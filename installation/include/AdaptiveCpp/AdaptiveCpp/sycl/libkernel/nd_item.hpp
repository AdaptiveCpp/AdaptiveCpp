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
#ifndef HIPSYCL_ND_ITEM_HPP
#define HIPSYCL_ND_ITEM_HPP

#include <functional>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "id.hpp"
#include "item.hpp"
#include "range.hpp"
#include "nd_range.hpp"
#include "multi_ptr.hpp"
#include "group.hpp"
#include "device_event.hpp"
#include "detail/mem_fence.hpp"

#include "detail/thread_hierarchy.hpp"
#include "detail/device_barrier.hpp"

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

template <int Dimensions = 1>
struct nd_item
{
  /* -- common interface members -- */
  static constexpr int dimensions = Dimensions;

  ACPP_KERNEL_TARGET
  id<Dimensions> get_global_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<Dimensions>() + (*_offset);
#else
    __acpp_if_target_sscp(return detail::get_global_id<Dimensions>() +
                                        (*_offset););
    return this->_global_id + (*_offset);
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_id<Dimensions>(dimension) + _offset->get(dimension);
#else
    __acpp_if_target_sscp(
        return detail::get_global_id<Dimensions>(dimension) +
                          _offset->get(dimension););
    return this->_global_id[dimension] + (*_offset)[dimension];
#endif
  }

  [[deprecated]]
  ACPP_KERNEL_TARGET
  size_t get_global(int dimension) const
  {
    return this->get_global_id(dimension);
  }

  [[deprecated]]
  ACPP_KERNEL_TARGET
  id<Dimensions> get_global() const
  {
    return this->get_global_id();
  }

  ACPP_KERNEL_TARGET
  size_t get_global_linear_id() const
  {
    __acpp_if_target_sscp(
        return __acpp_sscp_get_global_linear_id<Dimensions>(););

    return detail::linear_id<Dimensions>::get(get_global_id(),
                                              get_global_range());
  }

  ACPP_KERNEL_TARGET friend bool operator ==(const nd_item<Dimensions>& lhs, const nd_item<Dimensions>& rhs)
  {
    // nd_item is not allowed to be shared across work items, so comparison can only be true
    return true;
  }

  ACPP_KERNEL_TARGET friend bool operator !=(const nd_item<Dimensions>& lhs, const nd_item<Dimensions>& rhs)
  {
    return !(lhs==rhs);
  }

  [[deprecated]]
  ACPP_KERNEL_TARGET
  id<Dimensions> get_local() const
  {
    return this->get_local_id();
  }

  [[deprecated]] 
  ACPP_KERNEL_TARGET
  size_t get_local(int dimension) const
  {
    return this->get_local_id(dimension);
  }

  ACPP_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(dimension););

    return this->_local_id[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_local_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(););
    return this->_local_id;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_local_linear_id() const
  {
    __acpp_if_target_sscp(
        return __acpp_sscp_get_local_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(get_local_id(), get_local_range());
  }

  ACPP_KERNEL_TARGET
  group<Dimensions> get_group() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return group<Dimensions>{};
#else
    return group<Dimensions>{
        _group_id,
        _local_range,
        _num_groups,
        static_cast<detail::host_barrier_type *>(_group_barrier),
        get_local_id(),
        _local_memory_ptr};
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_group(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_group_id<Dimensions>(dimension););
    return this->_group_id[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::linear_id<Dimensions>::get(detail::get_group_id<Dimensions>(),
                                              detail::get_grid_size<Dimensions>());
#else
    __acpp_if_target_sscp(
        return __acpp_sscp_get_group_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(this->_group_id, this->_num_groups);
#endif
  }

  ACPP_KERNEL_TARGET
  sub_group get_sub_group() const
  {
    return sub_group{};
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_global_size<Dimensions>();)
    return this->_num_groups * this->_local_range;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_global_size<Dimensions>(dimension););
    return this->_num_groups[dimension] * this->_local_range[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(););
    return this->_local_range;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(dimension););
    return this->_local_range[dimension];
#endif
  }
  
  ACPP_KERNEL_TARGET
  range<Dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(););
    return this->_num_groups;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(dimension););
    return this->_num_groups[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_offset() const
  {
    return *_offset;
  }

  ACPP_KERNEL_TARGET
  nd_range<Dimensions> get_nd_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return nd_range<Dimensions>{detail::get_global_size<Dimensions>(),
                                detail::get_local_size<Dimensions>(),
                                get_offset()};
#else
    __acpp_if_target_sscp(return nd_range<Dimensions>{
        detail::get_global_size<Dimensions>(),
        detail::get_local_size<Dimensions>(), get_offset()};);
    
    return nd_range<Dimensions>{
      this->_num_groups * this->_local_range,
      this->_local_range,
      this->get_offset()
    };
#endif
  }

  HIPSYCL_LOOP_SPLIT_BARRIER ACPP_KERNEL_TARGET
  void barrier(access::fence_space space =
      access::fence_space::global_and_local) const
  {
    __acpp_if_target_device(
      detail::local_device_barrier(space);
    );
    __acpp_if_target_host(
        detail::host_barrier_type *barrier =
            static_cast<detail::host_barrier_type *>(_group_barrier);
        (*barrier)();
    );
  }

  template <access::mode accessMode = access::mode::read_write>
  ACPP_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    detail::mem_fence<accessMode>(accessSpace);
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
    return get_group().async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
    return get_group().async_work_group_copy(dest,
                                      src, numElements, srcStride);
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
    return get_group().async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename... eventTN>
  ACPP_KERNEL_TARGET
  void wait_for(eventTN... events) const
  {
    get_group().wait_for(events...);
  }

  
#if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  ACPP_KERNEL_TARGET
  nd_item(const id<Dimensions>* offset)
    : _offset{offset}
  {}
#else
  ACPP_KERNEL_TARGET
  nd_item(const id<Dimensions>* offset,
          id<Dimensions> group_id, id<Dimensions> local_id, 
          range<Dimensions> local_range, range<Dimensions> num_groups,
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
    __acpp_if_target_host(
      _group_barrier = static_cast<void*>(host_group_barrier);
    );
  }
#endif

private:
  const id<Dimensions>* _offset;

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  const id<Dimensions> _group_id;
  const id<Dimensions> _local_id;
  const range<Dimensions> _local_range;
  const range<Dimensions> _num_groups;
  const id<Dimensions> _global_id;
  void *_local_memory_ptr;
#endif

#ifndef SYCL_DEVICE_ONLY
  // Store void ptr to avoid function pointer types
  // appearing in SSCP code
  void* _group_barrier;
#endif
};

} // namespace sycl
} // namespace hipsycl

#endif
