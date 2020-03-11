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
#include "detail/mem_fence.hpp"

namespace hipsycl {
namespace sycl {

namespace vendor {
namespace hipsycl {
namespace synchronization {

struct none
{
  HIPSYCL_KERNEL_TARGET
  static void run() {}
};

template<access::fence_space Fence_space>
struct barrier
{
  HIPSYCL_KERNEL_TARGET
  static void run()
  {
#ifdef SYCL_DEVICE_ONLY
    __syncthreads();
#endif
  }
};

using local_barrier = barrier<access::fence_space::local_space>;

template <
  access::fence_space Fence_space,
  access::mode Mode = access::mode::read_write
>
struct mem_fence
{
  HIPSYCL_KERNEL_TARGET
  static void run()
  {
    detail::mem_fence<Fence_space, Mode>();
  }
};

using local_mem_fence = mem_fence<
  access::fence_space::local_space>;

using global_mem_fence = mem_fence<
  access::fence_space::global_space>;

using global_and_local_mem_fence = mem_fence<
  access::fence_space::global_and_local>;

}
}
}

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
    return detail::get_group_id<dimensions>(dimension);
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
    return detail::get_global_size<dimensions>(dimension);
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
    return detail::get_local_size<dimensions>(dimension);
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
    return detail::get_grid_size<dimensions>(dimension);
#else
    return _num_groups[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<dimensions>(dimension);
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

  template<
    typename Finalizer,
    typename workItemFunctionT>
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
#else
    iterate_over_work_items(_local_range, func);
#endif
    Finalizer::run();
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
    parallel_for_work_item<vendor::hipsycl::synchronization::local_barrier>(func);
  }

  template<
    typename Finalizer,
    typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              workItemFunctionT func) const
  {
#ifdef SYCL_DEVICE_ONLY
    parallelize_over_work_items(flexibleRange, func);
#else
    iterate_over_work_items(flexibleRange, func);
#endif
    Finalizer::run();
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(range<dimensions> flexibleRange,
                              workItemFunctionT func) const
  {
    parallel_for_work_item<vendor::hipsycl::synchronization::local_barrier>(
      flexibleRange, func);
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
    // in hipSYCL, we do not need to distinguish between global and local pointers,
    // so we can just call the async_work_group_copy() variant that has
    // global_ptr as dest and local_ptr as source.
    
    global_ptr<dataT> global_dest{dest.get()};
    local_ptr<dataT> local_src{src.get()};
    
    return async_work_group_copy(global_dest, local_src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
    
    for(size_t i = get_linear_local_id(); i < numElements; i += physical_local_size)
      dest[i] = src[i];
    __syncthreads();

#else
 #ifndef HIPCPU_NO_OPENMP
   #pragma omp simd
 #endif
   for(size_t i = 0; i < numElements; ++i)
      dest[i] = src[i];
#endif

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
    
    for(size_t i = get_linear_local_id(); i < numElements; i += physical_local_size)
      dest[i] = src[i * srcStride];
    __syncthreads();

#else
 #ifndef HIPCPU_NO_OPENMP
   #pragma omp simd
 #endif
   for(size_t i = 0; i < numElements; ++i)
      dest[i] = src[i * srcStride];
#endif

    return device_event{};
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
#ifdef SYCL_DEVICE_ONLY
    const size_t physical_local_size = get_local_range().size();
    
    for(size_t i = get_linear_local_id(); i < numElements; i += physical_local_size)
      dest[i * destStride] = src[i];
    __syncthreads();

#else
 #ifndef HIPCPU_NO_OPENMP
   #pragma omp simd
 #endif
   for(size_t i = 0; i < numElements; ++i)
      dest[i * destStride] = src[i];
#endif

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

#ifdef SYCL_DEVICE_ONLY
  size_t get_linear_local_id() const
  {
    return detail::linear_id<dimensions>::get(detail::get_local_id<dimensions>(),
                                              detail::get_local_size<dimensions>());
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallelize_over_work_items(const range<1> flexibleRange,
                                  workItemFunctionT&& func) const
  {
    const range<1> physical_range = this->get_local_range();
    for(size_t i = hipThreadIdx_x; i < flexibleRange.get(0); i += physical_range.get(0))
    {
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
      h_item<1> idx{id<1>{i}, flexibleRange};
  #else
      h_item<1> idx{id<1>{i}, flexibleRange, _group_id, _num_groups};
  #endif
      func(idx);
    }
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallelize_over_work_items(const range<2> flexibleRange,
                                  workItemFunctionT&& func) const
  {
    const range<2> physical_range = this->get_local_range();
    // Reverse dimensions of hipThreadIdx_* compared to flexibleRange.get()
    // to make sure that the fastest index in SYCL terminology is mapped
    // to the fastest index of the backend
    for(size_t i = hipThreadIdx_y; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = hipThreadIdx_x; j < flexibleRange.get(1); j += physical_range.get(1))
      {
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        h_item<2> idx{id<2>{i,j}, flexibleRange};
  #else
        h_item<2> idx{id<2>{i,j}, flexibleRange, _group_id, _num_groups};
  #endif
        func(idx);
      }
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallelize_over_work_items(const range<3> flexibleRange,
                                  workItemFunctionT&& func) const
  { 
    const range<3> physical_range = this->get_local_range();
    for(size_t i = hipThreadIdx_z; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = hipThreadIdx_y; j < flexibleRange.get(1); j += physical_range.get(1))
        for(size_t k = hipThreadIdx_x; k < flexibleRange.get(2); k += physical_range.get(2))
        {
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          h_item<3> idx{id<3>{i,j,k}, flexibleRange};
  #else
          h_item<3> idx{id<3>{i,j,k}, flexibleRange, _group_id, _num_groups};
  #endif
          func(idx);
        }
  }

#else
  template<typename workItemFunctionT>
  void iterate_over_work_items(const range<1> iteration_range,
                              workItemFunctionT&& func) const
  {
    #ifndef HIPCPU_NO_OPENMP
      #pragma omp simd 
    #endif
    for(size_t i = 0; i < iteration_range.get(0); ++i)
    {
      h_item<1> idx{
        id<1>{i}, 
        iteration_range, _group_id, _num_groups
      };

      func(idx);
    }
  // No memfence is needed here, because on CPU we only have one physical thread per work group.
  }


  template<typename workItemFunctionT>
  void iterate_over_work_items(const range<2> iteration_range,
                              workItemFunctionT&& func) const
  {
    for(size_t i = 0; i < iteration_range.get(0); ++i)
    #ifndef HIPCPU_NO_OPENMP
      #pragma omp simd 
    #endif
      for(size_t j = 0; j < iteration_range.get(1); ++j)
      {
        h_item<2> idx{
          id<2>{i,j}, 
          iteration_range, _group_id, _num_groups
        };

        func(idx);
      }
  // No memfence is needed here, because on CPU we only have one physical thread per work group.
  }

  template<typename workItemFunctionT>
  void iterate_over_work_items(const range<3> iteration_range,
                              workItemFunctionT&& func) const
  {
    for(size_t i = 0; i < iteration_range.get(0); ++i)
      for(size_t j = 0; j < iteration_range.get(1); ++j)
  #ifndef HIPCPU_NO_OPENMP
    #pragma omp simd 
  #endif
        for(size_t k = 0; k < iteration_range.get(2); ++k)
        {
          h_item<3> idx{
            id<3>{i,j,k}, 
            iteration_range, _group_id, _num_groups
          };

          func(idx);
        }
    // No memfence is needed here, because on CPU we only have one physical thread per work group.
  }

#endif // SYCL_DEVICE_ONLY

};

}
}

#endif
