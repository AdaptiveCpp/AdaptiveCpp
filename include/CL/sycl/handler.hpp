/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef SYCU_HANDLER_HPP
#define SYCU_HANDLER_HPP

#include "access.hpp"
#include "accessor.hpp"
#include "backend/backend.hpp"
#include "types.hpp"
#include "id.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "item.hpp"

namespace cl {
namespace sycl {

class queue;

class handler
{
  friend class queue;
  shared_ptr_class<queue> _queue;

  handler(const queue& q)
    : _queue{&q}
  {}

public:

  template <typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
  void require(accessor<dataT, dimensions, accessMode, accessTarget,
               access::placeholder::true_t> acc);

  //----- OpenCL interoperability interface is not supported
  /*

template <typename T>
void set_arg(int argIndex, T && arg);

template <typename... Ts>
void set_args(Ts &&... args);
*/
  //------ Kernel dispatch API


  template <typename KernelName, typename KernelType>
  void single_task(KernelType kernelFunc)
  {
    std::size_t shared_mem_size = 0;
    hipStream_t stream = _queue->get_hip_stream();

    single_task_kernel<<<1,1,shared_mem_size,stream>>>(kernelFunc);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc)
  {
    dispatch_kernel_without_offset(numWorkItems, kernelFunc);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, KernelType kernelFunc)
  {
    dispatch_kernel_with_offset(numWorkItems, workItemOffset, kernelFunc);
  }

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernelFunc)
  {
    dispatch_ndrange_kernel(executionRange, kernelFunc);
  }

  /*

  // Hierarchical kernel dispatch API - not yet supported

  template <typename KernelName, typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               WorkgroupFunctionType kernelFunc);

  template <typename KernelName, typename WorkgroupFunctionType, int dimensions>
  void parallel_for_work_group(range<dimensions> numWorkGroups,
                               range<dimensions> workGroupSize,
                               WorkgroupFunctionType kernelFunc);
  */

  /*
  void single_task(kernel syclKernel);

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems, kernel syclKernel);

  template <int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, kernel syclKernel);

  template <int dimensions>
  void parallel_for(nd_range<dimensions> ndRange, kernel syclKernel);
  */

  //------ Explicit copy operations API

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(accessor<T, dim, mode, tgt> src, shared_ptr_class<T> dest);

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(shared_ptr_class<T> src, accessor<T, dim, mode, tgt> dest);

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(accessor<T, dim, mode, tgt> src, T * dest);

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(const T * src, accessor<T, dim, mode, tgt> dest);

  template <typename T, int dim, access::mode mode, access::target tgt>
  void copy(accessor<T, dim, mode, tgt> src, accessor<T, dim, mode, tgt> dest);

  template <typename T, int dim, access::mode mode, access::target tgt>
  void update_host(accessor<T, dim, mode, tgt> acc);

  template<typename T, int dim, access::mode mode, access::target tgt>
  void fill(accessor<T, dim, mode, tgt> dest, const T& src);

private:

  template<int dimensions>
  dim3 get_default_local_range() const
  {
    if(dimensions == 1)
      return dim3(128);
    else if(dimensions == 2)
      return dim3(16,16);
    else if(dimensions == 3)
      return dim3(8,8,8);

    return dim3(1);
  }

  std::size_t ceil_division(std::size_t n,
                           std::size_t divisor) const
  {
    return (n + divisor - 1) / divisor;
  }

  template<int dimensions>
  void determine_grid_configuration(const range<dimensions>& num_work_items,
                                    dim3& grid,
                                    dim3& block) const
  {
    block = get_default_local_range<dimensions>();

    if(dimensions == 1)
      grid = dim3(ceil_division(num_work_items.get(0), block.x));
    else if (dimensions == 2)
      grid = dim3(ceil_division(num_work_items.get(0), block.x),
                  ceil_division(num_work_items.get(1), block.y));
    else if (dimensions == 3)
      grid = dim3(ceil_division(num_work_items.get(0), block.x),
                  ceil_division(num_work_items.get(1), block.y),
                  ceil_division(num_work_items.get(2), block.z));
    else
      grid = dim3(1);
  }

  template <typename KernelType, int dimensions>
  void dispatch_kernel_without_offset(range<dimensions> numWorkItems, KernelType kernelFunc)
  {
    std::size_t shared_mem_size = 0;
    hipStream_t stream = _queue->get_hip_stream();

    dim3 grid, block;
    determine_grid_configuration(numWorkItems, grid, block);

    parallel_for_kernel<<<grid,block,shared_mem_size,stream>>>(kernelFunc, numWorkItems);
  }

  template <typename KernelType, int dimensions>
  void dispatch_kernel_with_offset(range<dimensions> numWorkItems,
                                   id<dimensions> offset,
                                   KernelType kernelFunc)
  {
    std::size_t shared_mem_size = 0;
    hipStream_t stream = _queue->get_hip_stream();

    dim3 grid, block;
    determine_grid_configuration(numWorkItems, grid, block);

    parallel_for_kernel_with_offset<<<grid,block,shared_mem_size,stream>>>(kernelFunc,
                                                                           numWorkItems,
                                                                           offset);
  }

  template <typename KernelType, int dimensions>
  void dispatch_ndrange_kernel(nd_range<dimensions> execution_range,
                               KernelType kernelFunc)
  {

  }

  template<class Function>
  __global__ void single_task_kernel(Function f)
  {
    f();
  }

  template<int dimensions, class Function>
  __global__ void parallel_for_kernel(Function f,
                                      range<dimensions> execution_range)
  {
    if(detail::item_impl<dimensions>::get_linear_id() < execution_range.size())
      f(item<dimensions, false>{});
  }

  template<int dimensions, class Function>
  __global__ void parallel_for_kernel_with_offset(Function f,
                                                  range<dimensions> execution_range,
                                                  id<dimensions> offset)
  {
    if(detail::item_impl<dimensions>::get_linear_id() < execution_range.size())
      f(item<dimensions>{offset});
  }

  template<int dimensions, class Function>
  __global__ void parallel_for_ndrange_kernel(Function f)
  {

  }

};

} // namespace sycl
} // namespace cl

#endif
