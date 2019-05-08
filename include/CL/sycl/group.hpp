/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
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

#ifndef HIPSYCL_GROUP_HPP
#define HIPSYCL_GROUP_HPP

#include "id.hpp"
#include "range.hpp"
#include "access.hpp"
#include "device_event.hpp"
#include "backend/backend.hpp"
#include "detail/thread_hierarchy.hpp"
#include "multi_ptr.hpp"
#include "h_item.hpp"

namespace cl {
namespace sycl {

template <int dimensions = 1>
struct group
{

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_group_id<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<id<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_group_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  // Note: This returns the number of groups
  // in each dimension - the spec wrongly claims that it should
  // return the range "of the current group", i.e. the local range
  // which makes no sense.
  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_group_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_grid_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_grid_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_group_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_linear() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::linear_id<dimensions>::get(get_id(),
                                              get_group_range());
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
    h_item<dimensions> idx;
    func(idx);
    // We need implicit synchonization semantics
    mem_fence();
  }

  /// \todo Flexible ranges are currently unsupported.
  /*
  template<typename workItemFunctionT>
  __device__
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              workItemFunctionT func) const;
  */


  template <access::mode accessMode = access::mode::read_write>
  HIPSYCL_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    __syncthreads();
#else
    detail::invalid_host_call();
#endif
  }


  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const
  {
    size_t local_size = get_local_range().size();
    for(size_t i = get_linear(); i < numElements; i += local_size)
      dest[i] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
    size_t local_size = get_local_range().size();
    for(size_t i = get_linear(); i < numElements; i += local_size)
      dest[i] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
    size_t local_size = get_local_range().size();
    for(size_t i = get_linear(); i < numElements; i += local_size)
      dest[i] = src[i * srcStride];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
    size_t local_size = get_local_range().size();
    for(size_t i = get_linear(); i < numElements; i += local_size)
      dest[i * destStride] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename... eventTN>
  HIPSYCL_KERNEL_TARGET
  void wait_for(eventTN...) const {}

  //group(id<dimensions>* offset) = default;
};

}
}

#endif
