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
#ifndef HIPSYCL_HIPLIKE_KERNEL_LAUNCHER_HPP
#define HIPSYCL_HIPLIKE_KERNEL_LAUNCHER_HPP

#include <cassert>
#include <utility>
#include <cstdlib>
#include <string>

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"

#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                              \
    ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
 #define HIPSYCL_HIPLIKE_LAUNCHER_ALLOW_DEVICE_CODE
#endif


#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"

#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/dag_node.hpp"

#include "hipSYCL/glue/kernel_names.hpp"
#include "hipSYCL/glue/generic/code_object.hpp"


#if defined(HIPSYCL_HIPLIKE_LAUNCHER_ALLOW_DEVICE_CODE)

 #if !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
  #include "clang.hpp"
 #else
  #include "nvcxx.hpp"
 #endif
 
 #include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
 
#else

#ifndef __host__
 #define __host__
#endif
#ifndef __device__
 #define __device__
#endif
#ifndef __global__
 #define __global__
#endif
#ifndef __sycl_kernel
 #define __sycl_kernel
#endif
#ifndef __acpp_launch_integrated_kernel

#define __acpp_launch_integrated_kernel(f, grid, block, shared_mem, stream, \
                                           ...)                                \
  assert(false && "Dummy integrated kernel launch was called");

struct dim3 {
  dim3() = default;
  dim3(unsigned int x_, unsigned int y_, unsigned int z_)
  : x{x_}, y{y_}, z{z_} {}

  unsigned int x;
  unsigned int y;
  unsigned int z;
};

#endif
#endif

// Dummy kernel that is used in conjunction with 
// __builtin_get_device_side_stable_name() to extract kernel names
template<class KernelT>
__global__ void __acpp_kernel_name_template () {}

namespace hipsycl {
namespace glue {


namespace hiplike_dispatch {

template<int dimensions>
__host__ __device__
bool item_is_in_range(const sycl::item<dimensions, true>& item,
                      const sycl::range<dimensions>& execution_range,
                      const sycl::id<dimensions>& offset)
{
  for(int i = 0; i < dimensions; ++i)
  {
    if(item.get_id(i) >= offset.get(i) + execution_range.get(i))
    {
      return false;
    }
  }
  return true;
}

template<int dimensions>
__host__ __device__
bool item_is_in_range(const sycl::item<dimensions, false>& item,
                      const sycl::range<dimensions>& execution_range)
{
  for(int i = 0; i < dimensions; ++i)
  {
    if(item.get_id(i) >= execution_range.get(i))
    {
      return false;
    }
  }
  return true;
}

template<class F>
__host__ __device__
void device_invocation(F&& f)
{
  __acpp_if_target_device(
    f();
  )
}

template <typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f) {
  __acpp_if_target_device(
    device_invocation(f);
  );
}

template <typename KernelName, class Function, int dimensions>
__sycl_kernel void
parallel_for_kernel(Function f, sycl::range<dimensions> execution_range,
                    sycl::id<dimensions> offset, bool with_offset) {
  __acpp_if_target_device(
    // Note: We currently cannot have with_offset as template parameter
    // because this might cause clang to emit two kernels with the same
    // mangled name (variants with and without offset) if an explicit kernel
    // name is provided.
    if(with_offset) {
      device_invocation([&] __host__ __device__() {
        auto this_item = sycl::detail::make_item<dimensions>(
            sycl::detail::get_global_id<dimensions>() + offset, execution_range,
            offset);
        if (item_is_in_range(this_item, execution_range, offset))
          f(this_item);
      });
    } else {
      device_invocation([&] __host__ __device__() {
        auto this_item = sycl::detail::make_item<dimensions>(
            sycl::detail::get_global_id<dimensions>(), execution_range);
        if (item_is_in_range(this_item, execution_range))
          f(this_item);
      });
    }
  );
}

template <typename KernelName, class Function, int dimensions>
__sycl_kernel void parallel_for_ndrange_kernel(Function f,
                                               sycl::id<dimensions> offset) {
  __acpp_if_target_device(
    device_invocation(
      [&] __host__ __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        sycl::nd_item<dimensions> this_item{&offset};
#else
        sycl::nd_item<dimensions> this_item{
            &offset, sycl::detail::get_group_id<dimensions>(),
            sycl::detail::get_local_id<dimensions>(),
            sycl::detail::get_local_size<dimensions>(),
            sycl::detail::get_grid_size<dimensions>()};
#endif
        f(this_item);
      });
  );
}

template <typename KernelName, class Function, int dimensions>
__sycl_kernel void
parallel_for_workgroup(Function f,
                       // The logical group size is not yet used,
                       // but it's still useful to already have it here
                       // since it allows the compiler to infer 'dimensions'
                       sycl::range<dimensions> logical_group_size) {
  __acpp_if_target_device(
    device_invocation(
        [&] __host__ __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          sycl::group<dimensions> this_group;
#else
          sycl::group<dimensions> this_group{
              sycl::detail::get_group_id<dimensions>(),
              sycl::detail::get_local_size<dimensions>(),
              sycl::detail::get_grid_size<dimensions>()};
#endif
          f(this_group);
        });
  );
}

template<int DivisorX = 1, int DivisorY = 1, int DivisorZ = 1>
struct sp_multiversioning_properties {
  static constexpr int group_divisor_x = DivisorX;
  static constexpr int group_divisor_y = DivisorY;
  static constexpr int group_divisor_z = DivisorZ;

  template<int Dim>
  static constexpr auto get_sp_property_descriptor() {
    return sycl::detail::sp_property_descriptor<
        Dim, 0, decltype(get_hierarchical_decomposition<Dim>())>{};
  }

private:
  template<int Dim>
  static constexpr auto get_hierarchical_decomposition() {
    using namespace sycl::detail;
    using fallback_decomposition =
      nested_range<unknown_static_range, nested_range<static_range<1>>>;

    if constexpr(Dim == 1) {
      if constexpr(DivisorX % __acpp_warp_size == 0) {
        using decomposition =
            nested_range<unknown_static_range,
                         nested_range<static_range<__acpp_warp_size>>>;

        return decomposition{};
      } else {
        return fallback_decomposition{};
      }
    } else if constexpr(Dim == 2) {
      return fallback_decomposition{};
    } else {
      return fallback_decomposition{};
    }
  }
};

template <typename KernelName, class Function, class MultiversioningProps,
          int dimensions>
__sycl_kernel void
parallel_region(Function f, MultiversioningProps props,
                sycl::range<dimensions> num_groups,
                sycl::range<dimensions> group_size) {
  __acpp_if_target_device(
    device_invocation(
      [&] __host__ __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        sycl::group<dimensions> this_group;
#else
        sycl::group<dimensions> this_group{
            sycl::detail::get_group_id<dimensions>(),
            sycl::detail::get_local_size<dimensions>(),
            sycl::detail::get_grid_size<dimensions>()};
#endif
        using group_properties = std::decay_t<
            decltype(MultiversioningProps::template get_sp_property_descriptor<
                     dimensions>())>;
        f(sycl::detail::sp_group<group_properties>{this_group});
      });
  );
}

/// Flips dimensions such that the range is consistent with the mapping
/// of SYCL index dimensions to backend dimensions.
/// When launching a SYCL kernel, grid and blocksize should be transformed
/// using this function.
template<int dimensions>
inline dim3 make_kernel_launch_range(dim3 range);

template<>
inline dim3 make_kernel_launch_range<1>(dim3 range)
{
  return dim3(range.x, 1, 1);
}

template<>
inline dim3 make_kernel_launch_range<2>(dim3 range)
{
  return dim3(range.y, range.x, 1);
}

template<>
inline dim3 make_kernel_launch_range<3>(dim3 range)
{
  return dim3(range.z, range.y, range.x);
}

template <int dimensions>
inline dim3 make_kernel_launch_range(sycl::range<dimensions> r) {
  if(dimensions == 1)
    return make_kernel_launch_range<dimensions>(dim3(r[0], 1, 1));
  else if (dimensions == 2)
    return make_kernel_launch_range<dimensions>(dim3(r[0], r[1], 1));
  else if (dimensions == 3)
    return make_kernel_launch_range<dimensions>(dim3(r[0], r[1], r[2]));
  return dim3(1,1,1);
}

inline std::size_t ceil_division(std::size_t n, std::size_t divisor) {
  return (n + divisor - 1) / divisor;
}

template <int dimensions>
inline sycl::range<dimensions>
determine_grid_configuration(const sycl::range<dimensions> &num_work_items,
                             const sycl::range<dimensions> &local_range) {
  sycl::range<dimensions> res;
  for (int i = 0; i < dimensions; ++i)
    res[i] = ceil_division(num_work_items[i], local_range[i]);
  return res;
}


} // hiplike_dispatch

template<rt::backend_id Backend_id, class Queue_type>
class hiplike_kernel_launcher : public rt::backend_kernel_launcher
{
public:
#define __acpp_invoke_kernel(nodeptr, f, KernelNameT, KernelBodyT, grid,    \
                                block, shared_mem, stream, ...)                \
  if (false) {                                                                 \
    __acpp_kernel_name_template<KernelNameT><<<1, 1>>>();                   \
    __acpp_kernel_name_template<KernelBodyT><<<1, 1>>>();                   \
  }                                                                            \
  if constexpr (is_launch_from_module()) {                                     \
    invoke_from_module<KernelNameT, KernelBodyT>(nodeptr, grid, block,         \
                                                 shared_mem, __VA_ARGS__);     \
  } else {                                                                     \
    __acpp_launch_integrated_kernel(f, grid, block, shared_mem, stream,     \
                                       __VA_ARGS__)                            \
  }

  hiplike_kernel_launcher()
      : _queue{nullptr}, _invoker{[](rt::dag_node*) {}} {}

  virtual ~hiplike_kernel_launcher() {}

  virtual void set_params(void *q) override {
    _queue = reinterpret_cast<Queue_type*>(q);
  }

  Queue_type *get_queue() const {
    return _queue;
  }

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k) {
    
    this->_type = type;

    using kernel_name_t = typename KernelNameTraits::name;

    sycl::range<Dim> effective_local_range = local_range;
    if constexpr (type == rt::kernel_type::basic_parallel_for) {
      // If local range is non 0, we use it as a hint to override
      // the default selection
      if(local_range.size() == 0) {
        if constexpr (Dim == 1)
          effective_local_range = sycl::range<1>{128};
        else if constexpr (Dim == 2)
          effective_local_range = sycl::range<2>{16, 16};
        else if constexpr (Dim == 3)
          effective_local_range = sycl::range<3>{4, 8, 8};
      }
      HIPSYCL_DEBUG_INFO << "hiplike_kernel_launcher: Submitting high-level "
                            "parallel for with selected total group size of "
                         << effective_local_range.size() << std::endl;
    }

    _invoker = [=](rt::dag_node* node) mutable {
      assert(_queue != nullptr);
      
      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k);

      if constexpr (type == rt::kernel_type::single_task) {

        __acpp_invoke_kernel(
            node, hiplike_dispatch::single_task_kernel<kernel_name_t>,
            kernel_name_t, Kernel, dim3(1, 1, 1), dim3(1, 1, 1),
            dynamic_local_memory, _queue->get_native_type(), k);

      } else if constexpr (type == rt::kernel_type::custom) {
       
        sycl::interop_handle handle{_queue->get_device(),
                                    static_cast<void *>(_queue)};

        k(handle);

      } else {

        sycl::range<Dim> grid_range =
            hiplike_dispatch::determine_grid_configuration(
                global_range, effective_local_range);

        bool is_with_offset = false;
        for (std::size_t i = 0; i < Dim; ++i)
          if (offset[i] != 0)
            is_with_offset = true;

        int required_dynamic_local_mem =
            static_cast<int>(dynamic_local_memory);

        if constexpr (type == rt::kernel_type::basic_parallel_for) {

          __acpp_invoke_kernel(node,
              hiplike_dispatch::parallel_for_kernel<kernel_name_t>, kernel_name_t,
              Kernel,
              hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
              hiplike_dispatch::make_kernel_launch_range<Dim>(
                  effective_local_range),
              required_dynamic_local_mem, _queue->get_native_type(), k,
              global_range, offset, is_with_offset);

        } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

          for (int i = 0; i < Dim; ++i)
            assert(global_range[i] % effective_local_range[i] == 0);

          __acpp_invoke_kernel(node,
              hiplike_dispatch::parallel_for_ndrange_kernel<kernel_name_t>,
              kernel_name_t, Kernel,
              hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
              hiplike_dispatch::make_kernel_launch_range<Dim>(
                  effective_local_range),
              required_dynamic_local_mem, _queue->get_native_type(), k, offset);

        } else if constexpr (type ==
                              rt::kernel_type::hierarchical_parallel_for) {

          for (int i = 0; i < Dim; ++i)
            assert(global_range[i] % effective_local_range[i] == 0);

          __acpp_invoke_kernel(node,
              hiplike_dispatch::parallel_for_workgroup<kernel_name_t>,
              kernel_name_t, Kernel,
              hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
              hiplike_dispatch::make_kernel_launch_range<Dim>(
                  effective_local_range),
              required_dynamic_local_mem, _queue->get_native_type(), k,
              effective_local_range);

        } else if constexpr (type == rt::kernel_type::scoped_parallel_for) {

          for (int i = 0; i < Dim; ++i)
            assert(global_range[i] % effective_local_range[i] == 0);

          auto invoke_scoped_kernel = [&](auto multiversioning_props) {
            using multiversioned_parameters = decltype(multiversioning_props);

            using multiversioned_name_t =
                typename KernelNameTraits::template multiversioned_name<
                    multiversioned_parameters>;
            
            auto multiversioned_kernel_body =
                KernelNameTraits::template make_multiversioned_kernel_body<
                    multiversioned_parameters>(k);
            
            using sp_properties_t = decltype(multiversioning_props);

            __acpp_invoke_kernel(node,
                hiplike_dispatch::parallel_region<multiversioned_name_t>,
                multiversioned_name_t, decltype(multiversioned_kernel_body),
                hiplike_dispatch::make_kernel_launch_range<Dim>(grid_range),
                hiplike_dispatch::make_kernel_launch_range<Dim>(
                    effective_local_range),
                required_dynamic_local_mem, _queue->get_native_type(),
                multiversioned_kernel_body, multiversioning_props, grid_range,
                effective_local_range);
          };

          if constexpr(Dim == 1) {
            if(effective_local_range[0] % 64 == 0) {
              using sp_properties_t =
                  hiplike_dispatch::sp_multiversioning_properties<64>;
              invoke_scoped_kernel(sp_properties_t{});
            } else if(effective_local_range[0] % 32 == 0) {
              using sp_properties_t =
                  hiplike_dispatch::sp_multiversioning_properties<32>;
              invoke_scoped_kernel(sp_properties_t{});
            } else {
              using sp_properties_t =
                  hiplike_dispatch::sp_multiversioning_properties<1>;
              invoke_scoped_kernel(sp_properties_t{});
            }
          } else {
            using sp_properties_t =
                  hiplike_dispatch::sp_multiversioning_properties<1, 1, 1>;
              invoke_scoped_kernel(sp_properties_t{});
          }

        } else {
          assert(false && "Unsupported kernel type");
        }
      }
    };
  }

  virtual int get_backend_score(rt::backend_id b) const final override {
    return (b == Backend_id) ? 2 : -1;
  }

  virtual void invoke(rt::dag_node *node,
                      const rt::kernel_configuration &) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:
  
  static constexpr bool is_launch_from_module() {

    constexpr auto is_cuda_module_launch = [](){
#ifdef __ACPP_MULTIPASS_CUDA_HEADER__
      return Backend_id == rt::backend_id::cuda;
#else
      return false;
#endif
    };

    constexpr auto is_hip_module_launch = [](){
#ifdef __ACPP_MULTIPASS_HIP_HEADER__
      return Backend_id == rt::backend_id::hip;
#else
      return false;
#endif
    };

    return is_cuda_module_launch() || is_hip_module_launch();
  }

  template<class KernelT>
  std::string get_stable_kernel_name() const {
    // On clang 11, rely on __builtin_unique_stable_name()
#if defined(__clang__) && __clang_major__ == 11

    return __builtin_unique_stable_name(KernelT);

    // If we have builtin_get_device_side_mangled_name, rely on
    // __acpp_kernel_name_template unless we are in split compiler mode -
    // here we follow the traditional mangling based on the type.
    // In thas case, unnamed kernel lambdas are unsupported which is enforced
    // by the clang plugin in the device compilation pass.
#elif __has_builtin(__builtin_get_device_side_mangled_name) &&                 \
    !defined(__ACPP_SPLIT_COMPILER__)
    
    // The builtin unfortunately only works with __global__ or
    // __device__ functions. Since our kernel launchers cannot be __global__
    // when semantic analysis runs, we cannot apply the builtin
    // directly to our kernel launchers. Use dummy __global__ instead
    std::string name_template = __builtin_get_device_side_mangled_name(
      __acpp_kernel_name_template<KernelT>);
    std::string template_marker = "_Z27__acpp_kernel_name_template";
    std::string replacement = "_Z13__acpp_kernel";
    name_template.erase(0, template_marker.size());
    return replacement + name_template;
#else
    return typeid(KernelT).name();
#endif
  }

  template <class KernelName, class KernelBodyT, typename... Args>
  void invoke_from_module(rt::dag_node* node, dim3 grid_size, dim3 block_size,
                          unsigned dynamic_shared_mem, Args... args) {
    assert(node);
  
#if defined(__ACPP_MULTIPASS_CUDA_HEADER__) || defined(__ACPP_MULTIPASS_HIP_HEADER__)

    std::size_t local_hcf_object_id = 0;
#ifdef __ACPP_MULTIPASS_CUDA_HEADER__
    if(Backend_id == rt::backend_id::cuda) {
      local_hcf_object_id = __acpp_local_cuda_hcf_object_id;
    }
#endif
#ifdef __ACPP_MULTIPASS_HIP_HEADER__
    if(Backend_id == rt::backend_id::hip) {
      local_hcf_object_id = __acpp_local_hip_hcf_object_id;
    }
#endif

    std::array<void *, sizeof...(Args)> kernel_args{
      static_cast<void *>(&args)...
    };
    std::array<std::size_t, sizeof...(Args)> arg_sizes{
      sizeof(Args)...
    };

    std::string kernel_name_tag = get_stable_kernel_name<KernelName>();
    std::string kernel_body_name = get_stable_kernel_name<KernelBodyT>();

    assert(this->get_launch_capabilities().get_multipass_invoker());
    rt::multipass_code_object_invoker *invoker =
        this->get_launch_capabilities().get_multipass_invoker().value();

    assert(invoker &&
            "Runtime backend does not support invoking kernels from modules");

    auto num_groups = rt::range<3>{grid_size.x, grid_size.y, grid_size.z};
    auto group_size = rt::range<3>{block_size.x, block_size.y, block_size.z};

    const rt::kernel_operation &op =
        *static_cast<rt::kernel_operation*>(node->get_operation());

    rt::result err = invoker->submit_kernel(op, local_hcf_object_id,
        num_groups, group_size, dynamic_shared_mem, kernel_args.data(),
        arg_sizes.data(), kernel_args.size(), kernel_name_tag,
        kernel_body_name);

    if (!err.is_success())
      rt::register_error(err);
#else
    assert(false && "No module available to invoke kernels from");
#endif
  }

  Queue_type *_queue;
  rt::kernel_type _type;
  std::function<void (rt::dag_node*)> _invoker;
};

}
}

#undef __acpp_invoke_kernel
#undef __sycl_kernel

#endif
