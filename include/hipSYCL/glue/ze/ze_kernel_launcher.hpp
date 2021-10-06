/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_ZE_KERNEL_LAUNCHER_HPP
#define HIPSYCL_ZE_KERNEL_LAUNCHER_HPP


#include <cassert>
#include <tuple>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "hipSYCL/sycl/libkernel/detail/device_array.hpp"
#include "hipSYCL/sycl/interop_handle.hpp"
#include "hipSYCL/glue/generic/module.hpp"

#ifdef SYCL_DEVICE_ONLY
#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#endif

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

namespace hipsycl {
namespace glue {

namespace ze_dispatch {




#ifdef SYCL_DEVICE_ONLY
#define __sycl_kernel __attribute__((sycl_kernel))
#else
#define __sycl_kernel
#endif

// clang's SYCL frontend disassembles kernel lambdas and passes
// the actual lambda captures as kernel arguments. This is not
// required for hipSYCL because we do not use any opaque
// memory objects such as cl_mem and instead rely on buffers/accessors
// on top of USM pointers.
// The problem with the clang SYCL approach is that it requires either
// compiler support in the host pass to know the captures, or requires 
// using the clang SYCL generated integration header which does not work
// with the hipSYCL syclcc driver.
// To prevent clang from disassembling the kernel, we type-erase kernel 
// lambdas by putting their memory into array<uint, size>, and reinterpret
// this memory as lambda once we are inside the kernel. 
// Arrays are passed by clang into kernels by passing their individual
// elements as arguments. Because we know sizeof(KernelLambda), we can
// easily know how to pass arrays on the host side.
template<class KernelType>
class packed_kernel {
public:
  using component_type = uint32_t;

  packed_kernel() = default;
  packed_kernel(KernelType k) {
    init(reinterpret_cast<const unsigned char*>(&k));
  }

  template<typename... Args>
  void operator()(Args&&... args) const {
    const KernelType* k = reinterpret_cast<const KernelType*>(&(_data[0]));
    (*k)(args...);
  }

  constexpr std::size_t get_num_components() const {
    return num_components;
  }

  constexpr std::size_t get_component_size() const {
    return sizeof(component_type);
  }

  component_type* get_components() {
    return &(_data[0]);
  }

  const component_type* get_components() const {
    return &(_data[0]);
  }
private:

  void init(const unsigned char* kernel_bytes) {

    for(int i = 0; i < static_cast<int>(num_components); ++i) {
      union component {
        component_type c;
        unsigned char bytes[sizeof(component_type)];
      };

      component current_component;

      for(int byte = 0; byte < sizeof(component_type); ++byte) {
        std::size_t pos = i * sizeof(component_type) + byte;
        if(pos < kernel_size){
          current_component.bytes[byte] =
            kernel_bytes[pos];
        } else {
          current_component.bytes[byte] = 0;
        }
      }

      _data[i] = current_component.c;

    }
  }

  static constexpr std::size_t kernel_size = sizeof(KernelType);
  static constexpr std::size_t num_components =
    (kernel_size + sizeof(component_type)-1) / sizeof(component_type);

  using array_type = sycl::detail::device_array<component_type, num_components>;

  array_type _data;

  static_assert(sizeof(array_type) == sizeof(component_type) * num_components,
                "device_array size is invalid");
};

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_single_task(const packed_kernel<KernelType> &kernel) {
  kernel();
}

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_parallel_for(const packed_kernel<KernelType>& kernel) {
  kernel();
}

}

class ze_kernel_launcher : public rt::backend_kernel_launcher
{
public:
#ifdef SYCL_DEVICE_ONLY
#define __hipsycl_invoke_kernel(f, KernelNameT, KernelBodyT, num_groups,       \
                                group_size, local_mem, ...)                    \
  f(__VA_ARGS__);
#else
#define __hipsycl_invoke_kernel(f, KernelNameT, KernelBodyT, num_groups,       \
                                group_size, local_mem, ...)                    \
  invoke_from_module<KernelName, KernelBodyT>(num_groups, group_size,          \
                                              local_mem, __VA_ARGS__);
#endif

  ze_kernel_launcher() : _queue{nullptr}{}
  virtual ~ze_kernel_launcher(){}

  virtual void set_params(void* q) override {
    _queue = static_cast<rt::ze_queue*>(q);
  }

  template <class KernelNameTraits, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {

    using kernel_name_t = typename KernelNameTraits::suggested_mangling_name;

    this->_type = type;
    
    this->_invoker = [=](rt::dag_node* node) mutable {
      
      static_cast<rt::kernel_operation *>(node->get_operation())
          ->initialize_embedded_pointers(k, reductions...);

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
        HIPSYCL_DEBUG_INFO << "ze_kernel_launcher: Submitting high-level "
                              "parallel for with selected total group size of "
                          << effective_local_range.size() << std::endl;
      }

      sycl::range<Dim> num_groups;
      for(int i = 0; i < Dim; ++i) {
        num_groups[i] = (global_range[i] + effective_local_range[i] - 1) /
                        effective_local_range[i];
      }

      bool is_with_offset = false;
      for(int i = 0; i < Dim; ++i)
        if(offset[i] != 0)
          is_with_offset = true;

      if constexpr(type == rt::kernel_type::single_task){
        rt::range<3> single_item{1,1,1};

        __hipsycl_invoke_kernel(ze_dispatch::kernel_single_task<kernel_name_t>,
                                kernel_name_t, Kernel, single_item, single_item, 0,
                                ze_dispatch::packed_kernel{k});

      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

        auto kernel_wrapper = [global_range, k, offset, is_with_offset](){
#ifdef SYCL_DEVICE_ONLY
          sycl::id<Dim> gid = sycl::detail::get_global_id<Dim>();

          bool is_within_range = true;

          for(int i = 0; i < Dim; ++i)
            if(gid[i] >= global_range[i])
              is_within_range = false;

          if(is_within_range) {
            if(is_with_offset) {
              auto item = sycl::detail::make_item(gid + offset, 
                            global_range, offset);
              k(item);
            } else {
              auto item = sycl::detail::make_item(gid, global_range);
              k(item);
            }
          }
#else
          (void)k;
          (void)global_range;
          (void)offset;
          (void)is_with_offset;
#endif
        };

        __hipsycl_invoke_kernel(ze_dispatch::kernel_parallel_for<kernel_name_t>,
                                kernel_name_t, Kernel,
                                make_kernel_launch_range(num_groups),
                                make_kernel_launch_range(effective_local_range),
                                dynamic_local_memory, ze_dispatch::packed_kernel{kernel_wrapper});

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

        auto kernel_wrapper = [k, offset](){
#ifdef SYCL_DEVICE_ONLY
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          sycl::nd_item<Dim> this_item{&offset};
#else
          sycl::nd_item<Dim> this_item{
            &offset, sycl::detail::get_group_id<Dim>(),
            sycl::detail::get_local_id<Dim>(),
            sycl::detail::get_local_size<Dim>(),
            sycl::detail::get_grid_size<Dim>()
          };
#endif
          k(this_item);
#else
          (void)k;
          (void)offset;
#endif
        };

        __hipsycl_invoke_kernel(ze_dispatch::kernel_parallel_for<kernel_name_t>,
                                kernel_name_t, Kernel,
                                make_kernel_launch_range(num_groups),
                                make_kernel_launch_range(effective_local_range),
                                dynamic_local_memory, ze_dispatch::packed_kernel{kernel_wrapper});

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {
        rt::register_error(__hipsycl_here(), rt::error_info{
          "ze_kernel_launcher: hierarchical parallel for is not yet supported"});

      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

        auto kernel_wrapper = [k](){
#ifdef SYCL_DEVICE_ONLY

#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          sycl::group<Dim> this_group;
#else
          sycl::group<Dim> this_group{
            sycl::detail::get_group_id<Dim>(),
            sycl::detail::get_local_size<Dim>(),
            sycl::detail::get_grid_size<Dim>()};
#endif
          // TODO: We should actually query the subgroup size of the device
          // and then multiversion the kernel based on this. Currently,
          // scoped parallelism on SPIR-V just uses scalar subgroups.
          auto determine_group_properties = [](){
            if constexpr(Dim == 1) {
              return sycl::detail::sp_property_descriptor<Dim, 0, 1>{};
            } else if constexpr(Dim == 2){
              return sycl::detail::sp_property_descriptor<Dim, 0, 1, 1>{};
            } else {
              return sycl::detail::sp_property_descriptor<Dim, 0, 1, 1, 1>{};
            }
          };

          using group_properties = std::decay_t<decltype(determine_group_properties())>;

          k(sycl::detail::sp_group<group_properties>{this_group});
#else
          (void)k;
#endif
        };

        __hipsycl_invoke_kernel(ze_dispatch::kernel_parallel_for<kernel_name_t>,
                                kernel_name_t, Kernel,
                                make_kernel_launch_range(num_groups),
                                make_kernel_launch_range(effective_local_range),
                                dynamic_local_memory, ze_dispatch::packed_kernel{kernel_wrapper});


      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{
            rt::device_id{rt::backend_descriptor{rt::hardware_platform::level_zero,
                                                 rt::api_platform::level_zero},
                          0},
            static_cast<void*>(nullptr)};

        k(handle);
      }
      else {
        assert(false && "Unsupported kernel type");
      }
      
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::level_zero;
  }

  virtual void invoke(rt::dag_node* node) final override {
    _invoker(node);
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:
  template<int Dim>
  rt::range<3> make_kernel_launch_range(sycl::range<Dim> r) const {
    if constexpr(Dim == 1) {
      return rt::range<3>{r[0], 1, 1};
    } else if constexpr(Dim == 2) {
      return rt::range<3>{r[1], r[0], 1};
    } else {
      return rt::range<3>{r[2], r[1], r[0]};
    }
  }

  template <class KernelName, class KernelBodyT,
            class WrappedLambdaT>
  void invoke_from_module(rt::range<3> num_groups, rt::range<3> group_size,
                          unsigned dynamic_local_mem,
                          ze_dispatch::packed_kernel<WrappedLambdaT> kernel) {
    
    
#ifdef __HIPSYCL_MULTIPASS_SPIRV_HEADER__
#if !defined(__clang_major__) || __clang_major__ < 11
  #error Multipass compilation requires clang >= 11
#endif
    if (this_module::get_num_objects<rt::backend_id::level_zero>() == 0) {
      rt::register_error(
          __hipsycl_here(),
          rt::error_info{
              "hiplike_kernel_launcher: Cannot invoke SPIR-V kernel: No code "
              "objects present in this module."});
      return;
    }

    const std::string *kernel_image =
        this_module::get_code_object<rt::backend_id::level_zero>("spirv");
    assert(kernel_image && "Invalid kernel image object");

    std::array<void *, kernel.get_num_components()> kernel_args;
    std::array<std::size_t, kernel.get_num_components()> arg_sizes;

    for(std::size_t i = 0; i < kernel.get_num_components(); ++i) {
      arg_sizes[i] = kernel.get_component_size();
      kernel_args[i] = static_cast<void*>(kernel.get_components() + i);
    }

    std::string kernel_name_tag = __builtin_unique_stable_name(KernelName);
    std::string kernel_body_name = __builtin_unique_stable_name(KernelBodyT);

    rt::module_invoker *invoker = _queue->get_module_invoker();

    assert(invoker &&
            "Runtime backend does not support invoking kernels from modules");

    rt::result err = invoker->submit_kernel(
        this_module::get_module_id<rt::backend_id::level_zero>(), "spirv",
        kernel_image, num_groups, group_size, dynamic_local_mem,
        kernel_args.data(), arg_sizes.data(), kernel_args.size(), kernel_name_tag,
        kernel_body_name);

    if (!err.is_success())
      rt::register_error(err);
#else
    assert(false && "No module available to invoke kernels from");
#endif
  
  }

  std::function<void (rt::dag_node*)> _invoker;
  rt::kernel_type _type;
  rt::ze_queue* _queue;
};

}
}

#undef __hipsycl_invoke_kernel
#undef __sycl_kernel

#endif
