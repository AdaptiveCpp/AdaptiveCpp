
/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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
