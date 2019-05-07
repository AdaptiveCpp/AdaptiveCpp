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

#ifndef HIPSYCL_ND_ITEM_HPP
#define HIPSYCL_ND_ITEM_HPP

#include "id.hpp"
#include "item.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "multi_ptr.hpp"
#include "group.hpp"
#include "device_event.hpp"
#include "detail/thread_hierarchy.hpp"

namespace cl {
namespace sycl {


class handler;

template <int dimensions = 1>
struct nd_item
{
  /* -- common interface members -- */

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_global() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_id<dimensions>() + (*_offset);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_id(dimension) + _offset->get(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return get_global(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_linear_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::linear_id<dimensions>::get(get_global(), get_global_range());
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_local() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id<dimensions>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
    return get_local(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_id() const
  {
    return detail::linear_id<dimensions>::get(get_local(), get_local_range());
  }

  HIPSYCL_KERNEL_TARGET
  group<dimensions> get_group() const
  {
    return group<dimensions>{};
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_group_id(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::linear_id<dimensions>::get(detail::get_group_id<dimensions>(),
                                              detail::get_grid_size<dimensions>());
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_num_groups() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_grid_size<dimensions>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_num_groups(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_grid_size(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_size<dimensions>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size<dimensions>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_offset() const
  {
    return *_offset;
  }

  HIPSYCL_KERNEL_TARGET
  nd_range<dimensions> get_nd_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return nd_range<dimensions>{detail::get_global_size<dimensions>(),
                                detail::get_local_size<dimensions>(),
                                get_offset()};
#endif
  }

  HIPSYCL_KERNEL_TARGET
  void barrier(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    __syncthreads();
#endif
  }

  template <access::mode accessMode = access::mode::read_write>
  HIPSYCL_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    barrier(accessSpace);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
    return get_group().async_work_group_copy(dest,
                                      src, numElements, srcStride);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
    return get_group().async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename... eventTN>
  HIPSYCL_KERNEL_TARGET
  void wait_for(eventTN... events) const
  {
    get_group().wait_for(events...);
  }

  HIPSYCL_KERNEL_TARGET
  nd_item(id<dimensions>* offset)
    : _offset{offset}
  {}
private:
  const id<dimensions>* _offset;

};

} // namespace sycl
} // namespace cl

#endif
