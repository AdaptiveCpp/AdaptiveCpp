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

#include "../kernel_launcher.hpp"
#include "hip_queue.hpp"

namespace hipsycl {
namespace rt {

namespace device {

// These helper functions are necessary because we cannot
// call device functions directly from __sycl_kernel functions
// with the clang plugin, since they are initially treated
// as host functions.
// TODO: Find a nicer solution for that.
template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_global_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_global_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_local_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_local_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::id<dimensions> get_group_id_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_group_id<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::id<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::range<dimensions> get_local_size_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_local_size<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::range<dimensions>>();
#endif
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline sycl::range<dimensions> get_grid_size_helper()
{
#ifdef __HIPSYCL_DEVICE_CALLABLE__
  return detail::get_grid_size<dimensions>();
#else
  return detail::invalid_host_call_dummy_return<sycl::range<dimensions>>();
#endif
}

template<typename KernelName, class Function>
__sycl_kernel void single_task_kernel(Function f)
{
  f();
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel(Function f,
                        sycl::range<dimensions> execution_range)
{
  auto this_item = detail::make_item<dimensions>(
    get_global_id_helper<dimensions>(), execution_range);
  if(item_is_in_range(this_item, execution_range, sycl::id<dimensions>{}))
    f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_kernel_with_offset(Function f,
                                    sycl::range<dimensions> execution_range,
                                    sycl::id<dimensions> offset)
{
  auto this_item = detail::make_item<dimensions>(
    get_global_id_helper<dimensions>() + offset, execution_range, offset);
  if(item_is_in_range(this_item, execution_range, offset))
    f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel
void parallel_for_ndrange_kernel(Function f, sycl::id<dimensions> offset)
{
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
  nd_item<dimensions> this_item{&offset};
#else
  nd_item<dimensions> this_item{
    &offset, 
    get_group_id_helper<dimensions>(), 
    get_local_id_helper<dimensions>(),
    get_local_size_helper<dimensions>(),
    get_grid_size_helper<dimensions>()
  };
#endif
  f(this_item);
}

template<typename KernelName, class Function, int dimensions>
__sycl_kernel 
void parallel_for_workgroup(Function f,
                            // The logical group size is not yet used,
                            // but it's still useful to already have it here
                            // since it allows the compiler to infer 'dimensions'
                            sycl::range<dimensions> logical_group_size)
{
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
  group<dimensions> this_group;
#else
  group<dimensions> this_group{
    get_group_id_helper<dimensions>(), 
    get_local_size_helper<dimensions>(),
    get_grid_size_helper<dimensions>()
  };
#endif
  f(this_group);
}

} // device

class hip_kernel_launcher : public backend_kernel_launcher
{
public:
  
  hip_kernel_launcher(hip_queue *q, std::size_t local_mem)
      : _queue{q}, _local_mem{local_mem} {}

  template <class KernelName, kernel_type type, int Dim, class Kernel>
  void bind(sycl::id<Dim> offset, sycl::range<Dim> global_range,
            sycl::range<Dim> local_range, Kernel k) {

    _invoker = [=]() {
      if constexpr(type == kernel_type::single_task){
        __hipsycl_launch_kernel(q->get_stream());
      } else if constexpr (type == kernel_type::basic_parallel_for) {
      } else if constexpr (type == kernel_type::ndrange_parallel_for) {

      } else if constexpr (type == kernel_type::hierarchical_parallel_for) {

      } else {
        assert(false && "Unsupported kernel type");
      }
      
    }
  }

  virtual backend_id get_backend() const final override;
  virtual void invoke() final override;

private:
  hip_queue *_queue;
  std::size_t _local_mem;
  std::function<void ()> _invoker;
};

}
}

#endif
