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

namespace cl {
namespace sycl {

class queue;

class handler {
  friend class queue;
  shared_ptr_class<queue> _queue;

  handler(const queue& q);

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

  /*
  template <typename KernelName, typename KernelType>
  void single_task(KernelType kernelFunc);

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems, KernelType kernelFunc);

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(range<dimensions> numWorkItems,
                    id<dimensions> workItemOffset, KernelType kernelFunc);

  template <typename KernelName, typename KernelType, int dimensions>
  void parallel_for(nd_range<dimensions> executionRange, KernelType kernelFunc);

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


  template<class Function>
  __global__ void parallel_for_kernel(Function f)
  {

  }

};

} // namespace sycl
} // namespace cl

#endif
