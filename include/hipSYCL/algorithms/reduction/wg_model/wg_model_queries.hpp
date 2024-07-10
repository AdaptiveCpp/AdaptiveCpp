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
#ifndef HIPSYCL_REDUCTION_WG_MODEL_QUERIES_HPP
#define HIPSYCL_REDUCTION_WG_MODEL_QUERIES_HPP

#include "hipSYCL/sycl/libkernel/detail/thread_hierarchy.hpp"
#include "hipSYCL/sycl/libkernel/detail/data_layout.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"

namespace hipsycl::algorithms::reduction::wg_model {


inline std::size_t get_global_linear_id(sycl::id<1> idx) {
  return idx[0];
}

template<int Dim>
std::size_t get_global_linear_id(sycl::nd_item<Dim> idx) {
  return idx.get_global_linear_id();
}

template<int Dim>
std::size_t get_global_linear_id(sycl::group<Dim> grp) {
  auto global_range = grp.get_group_range() * grp.get_local_range();
  auto global_id = grp.get_local_id() + grp.get_local_range() * grp.get_group_id();
  return sycl::detail::linear_id<Dim>::get(global_id, global_range);
}

template<int Dim>
std::size_t get_group_linear_id(sycl::id<Dim> idx) {
  sycl::id<Dim> grp_id = sycl::detail::get_group_id<Dim>();
  sycl::range<Dim> grp_range = sycl::detail::get_grid_size<Dim>();
  return sycl::detail::linear_id<Dim>::get(grp_id, grp_range);
}

template<int Dim>
std::size_t get_group_linear_id(sycl::nd_item<Dim> idx) {
  return idx.get_group_linear_id();
}

template<int Dim>
std::size_t get_group_linear_id(sycl::group<Dim> grp) {
  return grp.get_group_linear_id();
}

template<int Dim>
std::size_t get_local_linear_id(sycl::id<Dim> idx) {
  sycl::id<Dim> local_id = sycl::detail::get_local_id<Dim>();
  sycl::range<Dim> local_range = sycl::detail::get_local_size<Dim>();
  return sycl::detail::linear_id<Dim>::get(local_id, local_range);
}


template<int Dim>
std::size_t get_local_linear_id(sycl::nd_item<Dim> idx) {
  return idx.get_local_linear_id();
}

template<int Dim>
std::size_t get_local_linear_id(sycl::group<Dim> idx) {
  return idx.get_local_linear_id();
}

}

#endif
