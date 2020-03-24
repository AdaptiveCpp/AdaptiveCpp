/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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

#ifndef HIPSYCL_HIP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_HIP_KERNEL_LAUNCHER_HPP

#include <cassert>

#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/range.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/nd_item.hpp"
#include "hipSYCL/sycl/group.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/hip/hip_queue.hpp"


namespace hipsycl {
namespace glue {

namespace hip_dispatch {

template<int dimensions, bool with_offset>
__device__
bool item_is_in_range(const sycl::item<dimensions, with_offset>& item,
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

template<class F>
__host__ __device__
void device_invocation(F f)
{
#ifdef SYCL_DEVICE_ONLY
  f();
#else
  assert(false && "Attempted to execute device code path on host!");
#endif
}

template<typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f)
{
  device_invocation(f);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel(Function f,
                        sycl::range<dimensions> execution_range)
{
  device_invocation([=] __device__ () {
    auto this_item = sycl::detail::make_item<dimensions>(
      sycl::detail::get_global_id<dimensions>(), execution_range);
    if(item_is_in_range(this_item, execution_range, sycl::id<dimensions>{}))
      f(this_item);
  });
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel_with_offset(Function f,
                                    sycl::range<dimensions> execution_range,
                                    sycl::id<dimensions> offset)
{
  device_invocation([=] __device__() {
    auto this_item = sycl::detail::make_item<dimensions>(
        sycl::detail::get_global_id<dimensions>() + offset, execution_range, offset);
    if(item_is_in_range(this_item, execution_range, offset))
      f(this_item);
  });
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel
void parallel_for_ndrange_kernel(Function f, sycl::id<dimensions> offset)
{
  device_invocation([=] __device__() {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    sycl::nd_item<dimensions> this_item{&offset};
#else
    sycl::nd_item<dimensions> this_item{
        &offset,
        sycl::detail::get_group_id<dimensions>(),
        sycl::detail::get_local_id<dimensions>(),
        sycl::detail::get_local_size<dimensions>(),
        sycl::detail::get_grid_size<dimensions>()};
#endif
    f(this_item);
  });
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_workgroup(Function f,
                            // The logical group size is not yet used,
                            // but it's still useful to already have it here
                            // since it allows the compiler to infer 'dimensions'
                            sycl::range<dimensions> logical_group_size)
{
  device_invocation([=] __device__() {
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
}

} // hip_dispatch

class hip_kernel_launcher : public rt::backend_kernel_launcher
{
public:
  
  hip_kernel_launcher()
      : _queue{nullptr}{}


  void set_params(rt::hip_queue *q) {
    _queue = q;
  }
  
  template <class KernelName, rt::kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, Kernel k) {

    _invoker = [=]() {
      assert(_queue != nullptr);

      if constexpr(type == rt::kernel_type::single_task){
        //__hipsycl_launch_kernel(_queue->get_stream());
      } else if constexpr (type == rt::kernel_type::basic_parallel_for) {
      } else if constexpr (type == rt::kernel_type::ndrange_parallel_for) {

      } else if constexpr (type == rt::kernel_type::hierarchical_parallel_for) {

      } else {
        assert(false && "Unsupported kernel type");
      }
      
    };
  }

  virtual rt::backend_id get_backend() const final override {
    return rt::backend_id::hip;
  }

  virtual void invoke() final override {
    _invoker();
  }

  virtual ~hip_kernel_launcher(){}
private:
  rt::hip_queue *_queue;
  std::function<void ()> _invoker;
};

}
}

#endif
