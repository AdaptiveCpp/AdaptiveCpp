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
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/item.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

namespace hipsycl {
namespace glue {

namespace ze_dispatch {


class auto_name {};

template <typename KernelName = auto_name, typename KernelType>
__attribute__((sycl_kernel)) void
kernel_single_task(const KernelType &kernelFunc) {
  kernelFunc();
}

template <typename KernelName, typename KernelType, int Dims>
__attribute__((sycl_kernel)) void
kernel_parallel_for(const KernelType &KernelFunc) {
  KernelFunc(sycl::id<Dims>{});
}
}

class ze_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  ze_kernel_launcher() {}
  virtual ~ze_kernel_launcher(){}

  virtual void set_params(void*) override {}

  template <class KernelName, rt::kernel_type type, int Dim, class Kernel,
            typename... Reductions>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, std::size_t dynamic_local_memory,
            Kernel k, Reductions... reductions) {

    this->_type = type;
    
    this->_invoker = [=]() {

      bool is_with_offset = false;
      for (std::size_t i = 0; i < Dim; ++i)
        if (offset[i] != 0)
          is_with_offset = true;

      auto get_grid_range = [&]() {
        for (int i = 0; i < Dim; ++i){
          if (global_range[i] % local_range[i] != 0) {
            rt::register_error(__hipsycl_here(),
                               rt::error_info{"ze_dispatch: global range is "
                                              "not divisible by local range"});
          }
        }

        return global_range / local_range;
      };

      if constexpr(type == rt::kernel_type::single_task){
      
      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {

      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

      } else if constexpr( type == rt::kernel_type::scoped_parallel_for) {

      } else if constexpr (type == rt::kernel_type::custom) {
        sycl::interop_handle handle{
            rt::device_id{rt::backend_descriptor{rt::hardware_platform::level_zero,
                                                 rt::api_platform::level_zero},
                          0},
            static_cast<void*>(nullptr)};

        // Need to perform additional copy to guarantee deferred_pointers/
        // accessors are initialized
        auto initialized_kernel_invoker = k;
        initialized_kernel_invoker(handle);
      }
      else {
        assert(false && "Unsupported kernel type");
      }
      
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::level_zero;
  }

  virtual void invoke() final override {
    _invoker();
  }

  virtual rt::kernel_type get_kernel_type() const final override {
    return _type;
  }

private:

  template <class KernelName, class KernelBodyT, typename... Args>
  void invoke_from_module(rt::range<3> num_groups, rt::range<3> group_size,
                          unsigned dynamic_local_mem, Args... args) {
    
    
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
        this_module::get_code_object<Backend_id>("spirv");
    assert(kernel_image && "Invalid kernel image object");

    std::array<void *, sizeof...(Args)> kernel_args{
      static_cast<void *>(&args)...
    };

    std::string kernel_name_tag = __builtin_unique_stable_name(KernelName);
    std::string kernel_body_name = __builtin_unique_stable_name(KernelBodyT);

    rt::module_invoker *invoker = _queue->get_module_invoker();

    assert(invoker &&
            "Runtime backend does not support invoking kernels from modules");

    rt::result err = invoker->submit_kernel(
        this_module::get_module_id<Backend_id>(), "spirv",
        kernel_image, num_groups, group_size, dynamic_local_mem,
        kernel_args.data(), kernel_args.size(), kernel_name_tag,
        kernel_body_name);

    if (!err.is_success())
      rt::register_error(err);
#else
    assert(false && "No module available to invoke kernels from");
#endif
  
  }

  std::function<void ()> _invoker;
  rt::kernel_type _type;
};

}
}

#endif
