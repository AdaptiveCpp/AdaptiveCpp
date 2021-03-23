/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
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

#include <functional>

#include "id.hpp"
#include "item.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "multi_ptr.hpp"
#include "group.hpp"
#include "device_event.hpp"
#include "detail/mem_fence.hpp"

#ifdef SYCL_DEVICE_ONLY
#include "detail/thread_hierarchy.hpp"
#include "detail/device_barrier.hpp"
#endif

namespace hipsycl {
namespace sycl {

namespace detail {
#ifdef SYCL_DEVICE_ONLY
using host_barrier_type = void;
#else
using host_barrier_type = std::function<void()>;
#endif
}

class handler;

template <int dimensions = 1>
struct nd_item
{
  /* -- common interface members -- */

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_global_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<dimensions>() + (*_offset);
#else
    return this->_global_id + (*_offset);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<dimensions>(dimension) + _offset->get(dimension);
#else
    return this->_global_id[dimension] + (*_offset)[dimension];
#endif
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  size_t get_global(int dimension) const
  {
    return this->get_global_id(dimension);
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_global() const
  {
    return this->get_global_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_linear_id() const
  {
    return detail::linear_id<dimensions>::get(get_global_id(), get_global_range());
  }

  HIPSYCL_KERNEL_TARGET friend bool operator ==(const nd_item<dimensions>& lhs, const nd_item<dimensions>& rhs)
  {
    #if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
      return  lhs._offset == rhs._offset;
    #else
      return  lhs._group_id == rhs._group_id &&
              lhs._offset == rhs._offset &&
              lhs._local_id == rhs._local_id &&
              lhs._global_id == rhs._global_id &&
              lhs._num_groups == rhs._num_groups &&
              lhs._local_range == rhs._local_range;
    #endif
  }

  HIPSYCL_KERNEL_TARGET friend bool operator !=(const nd_item<dimensions>& lhs, const nd_item<dimensions>& rhs)
  {
    return !(lhs==rhs);
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_local() const
  {
    return this->get_local_id();
  }

  [[deprecated]] 
  HIPSYCL_KERNEL_TARGET
  size_t get_local(int dimension) const
  {
    return this->get_local_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<dimensions>(dimension);
#else
    return this->_local_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_local_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<dimensions>();
#else
    return this->_local_id;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_id() const
  {
    return detail::linear_id<dimensions>::get(get_local_id(), get_local_range());
  }

  HIPSYCL_KERNEL_TARGET
  group<dimensions> get_group() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return group<dimensions>{};
#else
    return group<dimensions>{_group_id, _local_range, _num_groups, _group_barrier, get_local_id(), _local_memory_ptr};
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<dimensions>(dimension);
#else
    return this->_group_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::linear_id<dimensions>::get(detail::get_group_id<dimensions>(),
                                              detail::get_grid_size<dimensions>());
#else
    return detail::linear_id<dimensions>::get(this->_group_id, this->_num_groups);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  sub_group get_sub_group() const
  {
    return sub_group{};
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<dimensions>();
#else
    return this->_num_groups * this->_local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<dimensions>(dimension);
#else
    return this->_num_groups[dimension] * this->_local_range[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<dimensions>();
#else
    return this->_local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<dimensions>(dimension);
#else
    return this->_local_range[dimension];
#endif
  }
  
  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<dimensions>();
#else
    return this->_num_groups;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<dimensions>(dimension);
#else
    return this->_num_groups[dimension];
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
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return nd_range<dimensions>{detail::get_global_size<dimensions>(),
                                detail::get_local_size<dimensions>(),
                                get_offset()};
#else
    return nd_range<dimensions>{
      this->_num_groups * this->_local_range,
      this->_local_range,
      this->get_offset()
    };
#endif
  }

  HIPSYCL_KERNEL_TARGET
  void barrier(access::fence_space space =
      access::fence_space::global_and_local) const
  {
#ifdef SYCL_DEVICE_ONLY
    detail::local_device_barrier(space);
#else
    (*_group_barrier)();
#endif
  }

  template <access::mode accessMode = access::mode::read_write>
  HIPSYCL_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    detail::mem_fence<accessMode>(accessSpace);
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

  
#if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  HIPSYCL_KERNEL_TARGET
  nd_item(const id<dimensions>* offset)
    : _offset{offset}
  {}
#else
  HIPSYCL_KERNEL_TARGET
  nd_item(id<dimensions>* offset, 
          id<dimensions> group_id, id<dimensions> local_id, 
          range<dimensions> local_range, range<dimensions> num_groups,
          detail::host_barrier_type* host_group_barrier = nullptr,
          void* local_memory_ptr = nullptr)
    : _offset{offset}, 
      _group_id{group_id}, 
      _local_id{local_id}, 
      _local_range{local_range},
      _num_groups{num_groups},
      _global_id{group_id * local_range + local_id},
      _local_memory_ptr(local_memory_ptr)
  {
#ifndef SYCL_DEVICE_ONLY
    _group_barrier = host_group_barrier;
#endif
  }
#endif

private:
  const id<dimensions>* _offset;

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  const id<dimensions> _group_id;
  const id<dimensions> _local_id;
  const range<dimensions> _local_range;
  const range<dimensions> _num_groups;
  const id<dimensions> _global_id;
  void *_local_memory_ptr;
#endif

#ifndef SYCL_DEVICE_ONLY
  detail::host_barrier_type* _group_barrier;
#endif
};

} // namespace sycl
} // namespace hipsycl

#endif
