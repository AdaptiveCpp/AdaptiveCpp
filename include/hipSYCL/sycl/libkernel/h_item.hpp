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

#ifndef HIPSYCL_H_ITEM_HPP
#define HIPSYCL_H_ITEM_HPP

#include "item.hpp"

#ifdef SYCL_DEVICE_ONLY
#include "detail/thread_hierarchy.hpp"
#endif

namespace hipsycl {
namespace sycl {

template<int dimensions>
struct group;

template <int dimensions>
struct h_item
{
  friend struct group<dimensions>;

  HIPSYCL_KERNEL_TARGET
  h_item(){}
public:
  /* -- common interface members -- */

  /// \return The global id with respect to the parallel_for_work_group
  /// invocation. Flexlible local ranges are not taken into account.
  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_global() const
  {
    return detail::make_item<dimensions>(
      this->get_global_id(),
      this->get_global_range());
  }

  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_local() const
  {
    return get_logical_local();
  }

  /// \return The local id in the logical iteration space.
  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_logical_local() const
  {
    return detail::make_item<dimensions>(this->_logical_local_id,
                                         this->_logical_range);
  }

  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_physical_local() const
  {
    return detail::make_item<dimensions>(this->get_physical_local_id(),
                                         this->get_physical_local_range());
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<dimensions>() * this->_logical_range;
#else
    return this->_num_groups * this->_logical_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<dimensions>(dimension) * this->_logical_range[dimension];
#else
    return this->_num_groups[dimension] * this->_logical_range[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_global_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<dimensions>() * this->_logical_range + this->_logical_local_id;
#else
    return this->_group_id * this->_logical_range + this->_logical_local_id;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<dimensions>(dimension) * _logical_range[dimension] + _logical_local_id[dimension];
#else
    return _group_id[dimension] * _logical_range[dimension] + _logical_local_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET friend bool operator ==(const h_item<dimensions> lhs, const h_item<dimensions> rhs)
  {
  const range<dimensions> _num_groups;
  #if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
    return lhs._logical_local_id == rhs._logical_local_id && 
           lhs._logical_range == rhs._logical_range;
  #else
    return lhs._logical_local_id == rhs._logical_local_id &&
           lhs._logical_range == rhs._logical_range &&
           lhs._group_id == rhs._group_id &&
           lhs._num_groups == rhs.num_groups;
  #endif
  }

  HIPSYCL_KERNEL_TARGET friend bool operator !=(const h_item<dimensions> lhs, const h_item<dimensions> rhs)
  {
    return !(lhs==rhs);
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
    return get_logical_local_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
    return get_logical_local_range(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_local_id() const
  {
    return get_logical_local_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
    return get_logical_local_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_logical_local_range() const
  {
    return _logical_range;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const
  {
    return _logical_range[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_logical_local_id() const
  {
    return _logical_local_id;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_id(int dimension) const
  {
    return _logical_local_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_physical_local_range() const
  {
#ifdef SYCL_DEVICE_ONLY
    return detail::get_local_size<dimensions>();
#else
    range<dimensions> size;
    for(int i = 0; i < dimensions; ++i)
      size[i] = 1; 
    return size;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const
  {
#ifdef SYCL_DEVICE_ONLY
    return detail::get_local_size<dimensions>(dimension);
#else
    return 1;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_physical_local_id() const
  {
#ifdef SYCL_DEVICE_ONLY
    return detail::get_local_id<dimensions>();
#else
    id<dimensions> local_id;
    for(int i = 0; i < dimensions; ++i)
      local_id[i] = 0; 
    return local_id;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const
  {
#ifdef SYCL_DEVICE_ONLY
    return detail::get_local_id<dimensions>(dimension);
#else
    return 0;
#endif
  }

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  // TODO Make this private and group<> a friend
  h_item(id<dimensions> logical_local_id,
        range<dimensions> logical_range,
        id<dimensions> group_id,
        range<dimensions> num_groups)
    : _logical_local_id{logical_local_id},
      _logical_range{logical_range},
      _group_id{group_id},
      _num_groups{num_groups}
  {}
#else
  // TODO Make this private and group<> a friend
  h_item(id<dimensions> logical_local_id,
        range<dimensions> logical_range)
    : _logical_local_id{logical_local_id},
      _logical_range{logical_range}
  {}
#endif
private:
  // We do not really have to store both the physical and logical ids.
  // * On GPU, the physical size can be retrieved from __hipsycl_lid_x/y/z
  // * On CPU, we want to parallelize across the work groups and have (hopefully)
  //   vectorized loops over the work items, so the physical id is always 0.
  // The same reasoning holds for the local sizes:
  // * On GPU, we get the physical range from __hipsycl_lsize_x/y/z
  // * On CPU, the physical range is always 1.
  //
  // Note that for the support of flexible work group sizes,
  // we have to explicitly store logical ids/ranges even
  // if HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO is defined.
  const id<dimensions> _logical_local_id;
  const range<dimensions> _logical_range;

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  const id<dimensions> _group_id;
  const range<dimensions> _num_groups;
#endif
};

}
}

#endif
