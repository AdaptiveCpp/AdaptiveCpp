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
#ifndef HIPSYCL_H_ITEM_HPP
#define HIPSYCL_H_ITEM_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "item.hpp"

#include "detail/thread_hierarchy.hpp"


namespace hipsycl {
namespace sycl {

template<int Dimensions>
struct group;

template <int Dimensions>
struct h_item
{
  friend struct group<Dimensions>;

  ACPP_KERNEL_TARGET
  h_item(){}
public:
  /* -- common interface members -- */
  static constexpr int dimensions = Dimensions;

  /// \return The global id with respect to the parallel_for_work_group
  /// invocation. Flexlible local ranges are not taken into account.
  ACPP_KERNEL_TARGET
  item<Dimensions, false> get_global() const
  {
    return detail::make_item<Dimensions>(
      this->get_global_id(),
      this->get_global_range());
  }

  ACPP_KERNEL_TARGET
  item<Dimensions, false> get_local() const
  {
    return get_logical_local();
  }

  /// \return The local id in the logical iteration space.
  ACPP_KERNEL_TARGET
  item<Dimensions, false> get_logical_local() const
  {
    return detail::make_item<Dimensions>(this->_logical_local_id,
                                         this->_logical_range);
  }

  ACPP_KERNEL_TARGET
  item<Dimensions, false> get_physical_local() const
  {
    return detail::make_item<Dimensions>(this->get_physical_local_id(),
                                         this->get_physical_local_range());
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>() * this->_logical_range;
#else
    return this->_num_groups * this->_logical_range;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>(dimension) * this->_logical_range[dimension];
#else
    return this->_num_groups[dimension] * this->_logical_range[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_global_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>() * this->_logical_range + this->_logical_local_id;
#else
    return this->_group_id * this->_logical_range + this->_logical_local_id;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension) * _logical_range[dimension] + _logical_local_id[dimension];
#else
    return _group_id[dimension] * _logical_range[dimension] + _logical_local_id[dimension];
#endif
  }

  ACPP_KERNEL_TARGET friend bool operator ==(const h_item<Dimensions> lhs, const h_item<Dimensions> rhs)
  {
  const range<Dimensions> _num_groups;
  #if defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
    return lhs._logical_local_id == rhs._logical_local_id && 
           lhs._logical_range == rhs._logical_range;
  #else
    return lhs._logical_local_id == rhs._logical_local_id &&
           lhs._logical_range == rhs._logical_range &&
           lhs._group_id == rhs._group_id &&
           lhs._num_groups == rhs._num_groups;
  #endif
  }

  ACPP_KERNEL_TARGET friend bool operator !=(const h_item<Dimensions> lhs, const h_item<Dimensions> rhs)
  {
    return !(lhs==rhs);
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_local_range() const
  {
    return get_logical_local_range();
  }

  ACPP_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
    return get_logical_local_range(dimension);
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_local_id() const
  {
    return get_logical_local_id();
  }

  ACPP_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
    return get_logical_local_id(dimension);
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_logical_local_range() const
  {
    return _logical_range;
  }

  ACPP_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const
  {
    return _logical_range[dimension];
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_logical_local_id() const
  {
    return _logical_local_id;
  }

  ACPP_KERNEL_TARGET
  size_t get_logical_local_id(int dimension) const
  {
    return _logical_local_id[dimension];
  }

  ACPP_KERNEL_TARGET
  range<Dimensions> get_physical_local_range() const
  {
    __acpp_if_target_device(
      return detail::get_local_size<Dimensions>();
    );
    __acpp_if_target_host(
      range<Dimensions> size;
      for(int i = 0; i < Dimensions; ++i)
        size[i] = 1; 
      return size;
    );

  }

  ACPP_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const
  {
    __acpp_if_target_device(
      return detail::get_local_size<Dimensions>(dimension);
    );
    __acpp_if_target_host(return 1;);
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_physical_local_id() const
  {
    __acpp_if_target_device(
      return detail::get_local_id<Dimensions>();
    );
    __acpp_if_target_host(
      id<Dimensions> local_id;
      for(int i = 0; i < Dimensions; ++i)
        local_id[i] = 0; 
      return local_id;
    );
  }

  ACPP_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const
  {
    __acpp_if_target_device(
      return detail::get_local_id<Dimensions>(dimension);
    );
    __acpp_if_target_host(return 0;);
  }

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  // TODO Make this private and group<> a friend
  h_item(id<Dimensions> logical_local_id,
        range<Dimensions> logical_range,
        id<Dimensions> group_id,
        range<Dimensions> num_groups)
    : _logical_local_id{logical_local_id},
      _logical_range{logical_range},
      _group_id{group_id},
      _num_groups{num_groups}
  {}
#else
  // TODO Make this private and group<> a friend
  h_item(id<Dimensions> logical_local_id,
        range<Dimensions> logical_range)
    : _logical_local_id{logical_local_id},
      _logical_range{logical_range}
  {}
#endif
private:
  // We do not really have to store both the physical and logical ids.
  // * On GPU, the physical size can be retrieved from __acpp_lid_x/y/z
  // * On CPU, we want to parallelize across the work groups and have (hopefully)
  //   vectorized loops over the work items, so the physical id is always 0.
  // The same reasoning holds for the local sizes:
  // * On GPU, we get the physical range from __acpp_lsize_x/y/z
  // * On CPU, the physical range is always 1.
  //
  // Note that for the support of flexible work group sizes,
  // we have to explicitly store logical ids/ranges even
  // if HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO is defined.
  const id<Dimensions> _logical_local_id;
  const range<Dimensions> _logical_range;

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  const id<Dimensions> _group_id;
  const range<Dimensions> _num_groups;
#endif
};

}
}

#endif
