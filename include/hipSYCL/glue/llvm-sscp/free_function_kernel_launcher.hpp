/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_FREE_FUNCTION_KERNEL_LAUNCHER_HPP
#define HIPSYCL_FREE_FUNCTION_KERNEL_LAUNCHER_HPP

#include "hipSYCL/glue/kernel_launcher_data.hpp"
#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_type.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// Name-extraction intrinsic for free function kernels.
//
// The SSCP compiler resolves this call (annotation "acpp_sscp_extract_kernel_name",
// see HostKernelNameExtractionPass) by writing the mangled symbol name of the
// function passed as the first argument into the global char array pointed to by
// the second argument.
//
// The parameter type is templated on the exact free function kernel signature so
// that the function pointer is passed *directly*, with no bitcast/indirection --
// the pass requires operand 0 to be an llvm::Function.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wundefined-internal"
template <class... Params>
[[clang::annotate("acpp_sscp_extract_kernel_name")]]
void __acpp_sscp_extract_free_function_kernel_name(void (*Func)(Params...),
                                                   const char *target);
#pragma clang diagnostic pop

namespace hipsycl {
namespace glue {
namespace free_function_kernels {

// Extract the (compiler-provided) mangled kernel name for the free function
// kernel Func. The result is cached in a per-Func static buffer whose contents
// are filled in by the SSCP HostKernelNameExtractionPass.
template <auto *Func>
const char *get_kernel_name() {
  // The compiler will resize this array to the kernel name length and write the
  // mangled name into it.
  static char __acpp_sscp_kernel_name[] = "kernel-name-extraction-failed";

  __acpp_sscp_extract_free_function_kernel_name(Func, &__acpp_sscp_kernel_name[0]);
  return &__acpp_sscp_kernel_name[0];
}

// Deduce the parameter types of a free function kernel from its type.
template <class F> struct function_parameter_types;
template <class... Params>
struct function_parameter_types<void (*)(Params...)> {
  using type = std::tuple<Params...>;
};

template <int Dim>
rt::range<3> flip_range(const sycl::range<Dim> &r) {
  rt::range<3> rt_range{1, 1, 1};
  for (int i = 0; i < Dim; ++i)
    rt_range[i] = r[Dim - i - 1];
  return rt_range;
}

// Append the raw bytes of a value to the contiguous argument blob and record
// its size.
inline void append_argument(std::vector<uint8_t> &blob,
                            std::vector<std::size_t> &sizes, const void *data,
                            std::size_t size) {
  const uint8_t *bytes = static_cast<const uint8_t *>(data);
  blob.insert(blob.end(), bytes, bytes + size);
  sizes.push_back(size);
}

// Convert an argument to the corresponding free function kernel parameter type
// and append its bytes. Converting to the *parameter* type (rather than packing
// the caller's argument type) guarantees the packed layout matches the kernel
// ABI even when implicit conversions are involved (e.g. 0 -> float* or int ->
// float).
template <class ParamType, class Arg>
void pack_one(std::vector<uint8_t> &blob, std::vector<std::size_t> &sizes,
              Arg &&arg) {
  ParamType converted = static_cast<ParamType>(std::forward<Arg>(arg));
  append_argument(blob, sizes, &converted, sizeof(ParamType));
}

template <auto *Func, std::size_t... I, class ArgTuple>
void pack_arguments(std::vector<uint8_t> &blob, std::vector<std::size_t> &sizes,
                    std::index_sequence<I...>, ArgTuple &&args) {
  using param_types = typename function_parameter_types<decltype(Func)>::type;
  (pack_one<std::tuple_element_t<I, param_types>>(
       blob, sizes, std::get<I>(std::forward<ArgTuple>(args))),
   ...);
}

// Deferred launch: executed by the runtime once the kernel's DAG node is
// scheduled. Reconstructs the per-argument pointers from the contiguous blob and
// submits the pre-compiled kernel identified by name from the code object.
inline rt::result
invoke(const kernel_launcher_data &cfg, rt::dag_node *node,
       const rt::kernel_configuration &kernel_config,
       const rt::backend_kernel_launch_capabilities &launch_capabilities,
       void *backend_params) {
  assert(node);
  auto *kernel_op = static_cast<rt::kernel_operation *>(node->get_operation());

  auto sscp_invoker = launch_capabilities.get_sscp_invoker();
  if (!sscp_invoker) {
    return rt::make_error(
        __acpp_here(),
        rt::error_info{"free_function_kernel_launcher: backend did not "
                       "configure the kernel launcher for SSCP."});
  }
  auto *invoker = sscp_invoker.value();

  auto selected_group_size = cfg.group_size;
  if (cfg.group_size.size() == 0)
    selected_group_size =
        invoker->select_group_size(cfg.global_size, cfg.group_size);

  rt::range<3> num_groups;
  for (int i = 0; i < 3; ++i) {
    num_groups[i] = (cfg.global_size[i] + selected_group_size[i] - 1) /
                    selected_group_size[i];
  }

  // Free function kernels forbid non-device-copyable arguments (e.g.
  // accessors), so there are no embedded pointers to initialize.

  const std::size_t num_args = cfg.sscp_argument_sizes.size();
  std::vector<void *> arg_pointers(num_args);
  std::size_t offset = 0;
  for (std::size_t i = 0; i < num_args; ++i) {
    arg_pointers[i] = cfg.kernel_args.data() + offset;
    offset += cfg.sscp_argument_sizes[i];
  }

  return invoker->submit_kernel(
      *kernel_op, cfg.sscp_hcf_object_id, num_groups, selected_group_size,
      cfg.local_mem_size, arg_pointers.data(),
      const_cast<std::size_t *>(cfg.sscp_argument_sizes.data()), num_args,
      std::string_view{cfg.sscp_kernel_id}, cfg.kernel_info, kernel_config);
}

// Populate a kernel_launcher_data for a free function kernel launch.
template <auto *Func, int Dim, typename... Args>
void configure(kernel_launcher_data &data, sycl::range<Dim> global_range,
               sycl::range<Dim> local_range, std::size_t local_mem_size,
               Args &&...args) {
  data.type = rt::kernel_type::ndrange_parallel_for;
  data.sscp_invoker = &invoke;

  data.global_size = flip_range(global_range);
  data.group_size = flip_range(local_range);
  data.local_mem_size = static_cast<unsigned>(local_mem_size);

  data.sscp_kernel_id = get_kernel_name<Func>();

  // Resolve the kernel by name. The free function kernel may be defined and
  // outlined in a different translation unit than this launch, so we cannot
  // rely on the local HCF object id. We first try the local object (the common
  // case, cheapest), then fall back to an object-independent lookup.
  data.sscp_hcf_object_id = __acpp_local_sscp_hcf_object_id;
  data.kernel_info = rt::hcf_cache::get().get_kernel_info(
      data.sscp_hcf_object_id, std::string_view{data.sscp_kernel_id});
  if (!data.kernel_info) {
    auto resolved = rt::hcf_cache::get().get_kernel_info_by_name(
        std::string_view{data.sscp_kernel_id});
    data.sscp_hcf_object_id = resolved.object_id;
    data.kernel_info = resolved.info;
  }

  data.kernel_args.clear();
  data.sscp_argument_sizes.clear();
  pack_arguments<Func>(data.kernel_args, data.sscp_argument_sizes,
                       std::make_index_sequence<sizeof...(Args)>{},
                       std::forward_as_tuple(std::forward<Args>(args)...));
}

} // namespace free_function_kernels
} // namespace glue
} // namespace hipsycl

#endif
