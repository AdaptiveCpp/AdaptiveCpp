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
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<dimensions>();
#else
    return _group_id;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id(dimension);
#else
    return _group_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<dimensions>();
#else
    return _num_groups * _local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size(dimension);
#else
    return _num_groups[dimension] * _local_range[dimension];
#endif
  }

  /// \return The physical local range for flexible work group sizes,
  /// the logical local range otherwise.
  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<dimensions>();
#else
    return _local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size(dimension);
#else
    return _local_range[dimension];
#endif
  }

  // Note: This returns the number of groups
  // in each dimension - earler versions of the spec wrongly 
  // claim that it should return the range "of the current group", 
  // i.e. the local range which makes no sense.
  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<dimensions>();
#else
    return _num_groups;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size(dimension);
#else
    return _num_groups[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id(dimension);
#else
    return _group_id[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_linear() const
  {
    return detail::linear_id<dimensions>::get(get_id(),
                                              get_group_range());
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
#ifdef SYCL_DEVICE_ONLY
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    h_item<dimensions> idx{detail::get_local_id<dimensions>(), detail::get_local_size<dimensions>()};
  #else
    h_item<dimensions> idx{detail::get_local_id<dimensions>(), _local_range, _group_id, _num_groups};
  #endif
    func(idx);
    // We need implicit synchonization semantics
    mem_fence();
#else
    const range<3> range3d = detail::range::range_cast<3>(_local_range);

  #ifndef HIPCPU_NO_OPENMP
    #pragma omp simd
  #endif
    for(size_t i = 0; i < range3d.get(0); ++i)
      for(size_t j = 0; j < range3d.get(1); ++j)
        for(size_t k = 0; k < range3d.get(2); ++k)
        {
          h_item<dimensions> idx{
            detail::id::construct_from_first_n<dimensions>(i,j,k),
            _local_range, _group_id, _num_groups
          };
          func(idx);
        }
    // No memfence is needed here, because on CPU we only have one physical thread per work group.
#endif
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              workItemFunctionT func) const
  {
    const range<3> logical_range3d = detail::range::range_cast<3>(flexibleRange);

#ifdef SYCL_DEVICE_ONLY
    const range<3> physical_range3d = detail::range::range_cast<3>(this->get_local_range());
    for(size_t i = 0; i < logical_range3d.get(0); i += physical_range3d.get(0))
      for(size_t j = 0; j < logical_range3d.get(1); j += physical_range3d.get(1))
        for(size_t k = 0; k < logical_range3d.get(2); k += physical_range3d.get(2))
        {
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          h_item<dimensions> idx{
            detail::id::construct_from_first_n<dimensions>(i,j,k), flexibleRange};
  #else
          h_item<dimensions> idx{
            detail::id::construct_from_first_n<dimensions>(i,j,k), 
            flexibleRange, _group_id, _num_groups
          };
  #endif
          func(idx);
        }
    mem_fence();
#else
  #ifndef HIPCPU_NO_OPENMP
    #pragma omp simd 
  #endif
    for(size_t i = 0; i < logical_range3d.get(0); ++i)
      for(size_t j = 0; j < logical_range3d.get(1); ++j)
        for(size_t k = 0; k < logical_range3d.get(2); ++k)
        {
          h_item<dimensions> idx{
            detail::id::construct_from_first_n<dimensions>(i,j,k), 
            flexibleRange, _group_id, _num_groups
          };

          func(idx);
        }
    // No memfence is needed here, because on CPU we only have one physical thread per work group.
#endif
  }


  template <access::mode accessMode = access::mode::read_write>
  HIPSYCL_KERNEL_TARGET
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    // On CPU, mem_fence() can be skipped since there is only one physical thread
    // per work group
#ifdef SYCL_DEVICE_ONLY
    __syncthreads();
#endif
  }


  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
#else
    const size_t physical_local_size = 1;
#endif

#ifndef HIPCPU_NO_OPENMP
  #pragma omp simd
#endif
    for(size_t i = get_linear(); i < numElements; i += physical_local_size)
      dest[i] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
#else
    const size_t physical_local_size = 1;
#endif

#ifndef HIPCPU_NO_OPENMP
  #pragma omp simd
#endif
    for(size_t i = get_linear(); i < numElements; i += physical_local_size)
      dest[i] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
#else
    const size_t physical_local_size = 1;
#endif

#ifndef HIPCPU_NO_OPENMP
  #pragma omp simd
#endif
    for(size_t i = get_linear(); i < numElements; i += physical_local_size)
      dest[i] = src[i * srcStride];
    mem_fence();

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
#else
    const size_t physical_local_size = 1;
#endif

#ifndef HIPCPU_NO_OPENMP
  #pragma omp simd
#endif
    for(size_t i = get_linear(); i < numElements; i += physical_local_size)
      dest[i * destStride] = src[i];
    mem_fence();

    return device_event{};
  }

  template <typename... eventTN>
  HIPSYCL_KERNEL_TARGET
  void wait_for(eventTN...) const {}

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  group(id<dimensions> group_id,
        range<dimensions> local_range,
        range<dimensions> num_groups)
  : _group_id{group_id}, 
    _local_range{local_range}, 
    _num_groups{num_groups}
  {}

private:
  const id<dimensions> _group_id;
  const range<dimensions> _local_range;
  const range<dimensions> _num_groups;
#endif
};

}
}

#endif
