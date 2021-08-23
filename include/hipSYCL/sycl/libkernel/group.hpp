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

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"

#include "id.hpp"
#include "range.hpp"
#include "device_event.hpp"
#include "multi_ptr.hpp"
#include "h_item.hpp"
#include "detail/mem_fence.hpp"
#include "sub_group.hpp"
#include "sp_item.hpp"
#include "memory.hpp"

#ifdef SYCL_DEVICE_ONLY
#include "detail/thread_hierarchy.hpp"
#include "detail/device_barrier.hpp"
#endif

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
    detail::local_device_barrier(Fence_space);
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

template <int Dimensions = 1>
struct group
{

  using id_type = id<Dimensions>;
  using range_type = range<Dimensions>;
  using linear_id_type = size_t;
  static constexpr int dimensions = Dimensions;
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_group_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>();
#else
    return _group_id;
#endif
  }

  [[deprecated("To get the work group id use get_group_id() in SYCL 2020")]]
  HIPSYCL_KERNEL_TARGET
  id<Dimensions> get_id() const
  {
    return get_group_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    return _group_id[dimension];
#endif
  }

  [[deprecated("To get the work group id use get_group_id(int) in SYCL 2020")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_id(int dimension) const
  {
    return get_group_id(dimension);
  }

  [[deprecated("get_global_range() doesn't exist in SYCL 2020 anymore")]]
  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>();
#else
    return _num_groups * _local_range;
#endif
  }

  [[deprecated("get_global_range(int) doesn't exist in SYCL 2020 anymore")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>(dimension);
#else
    return _num_groups[dimension] * _local_range[dimension];
#endif
  }

  /// \return The physical local range for flexible work group sizes,
  /// the logical local range otherwise.
  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>();
#else
    return _local_range;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>(dimension);
#else
    return _local_range[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_range() const
  {
    return get_local_range().size();
  }

  // Note: This returns the number of groups
  // in each dimension - earler versions of the spec wrongly 
  // claim that it should return the range "of the current group", 
  // i.e. the local range which makes no sense.
  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>();
#else
    return _num_groups;
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>(dimension);
#else
    return _num_groups[dimension];
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_range() const
  {
    return get_group_range().size();
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    return _group_id[dimension];
#endif
  }

  friend bool operator==(const group<Dimensions>& lhs, const group<Dimensions>& rhs){
    return lhs._group_id == rhs._group_id &&
           lhs._local_range == rhs._local_range &&
           lhs._num_groups == rhs._num_groups;
  }

  friend bool operator!=(const group<Dimensions>& lhs, const group<Dimensions>& rhs){
    return !(lhs == rhs);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
    return detail::linear_id<Dimensions>::get(get_id(),
                                              get_group_range());
  }

  [[deprecated("Use get_group_linear_id() instead.")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_linear() const
  {
    return get_group_linear_id();
  }

#ifdef SYCL_DEVICE_ONLY
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const
  {
    return detail::get_local_id<Dimensions>();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
    return detail::get_local_id<Dimensions>(dimension);
  }
#endif

  HIPSYCL_KERNEL_TARGET
  range<Dimensions> get_max_local_range() const{
    if constexpr (Dimensions == 1) {
      return {1024};
    } else if constexpr (Dimensions == 2) {
      return {1024, 1024};
    } else if constexpr (Dimensions == 3) {
      return {1024, 1024, 1024};
    } else {
      static_assert(std::is_same_v<int, int>, "Only three dimensional ranges are supported!");
    }
  }

  template<
    typename Finalizer,
    typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
#ifdef SYCL_DEVICE_ONLY
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    h_item<Dimensions> idx{detail::get_local_id<Dimensions>(), detail::get_local_size<Dimensions>()};
  #else
    h_item<Dimensions> idx{detail::get_local_id<Dimensions>(), _local_range, _group_id, _num_groups};
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
  void parallel_for_work_item(range<Dimensions> flexibleRange,
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
  void parallel_for_work_item(range<Dimensions> flexibleRange,
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
    
    for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
      dest[i] = src[i];
    detail::local_device_barrier(access::fence_space::global_and_local);

#else

#ifdef _OPENMP
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
    
    for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
      dest[i] = src[i * srcStride];
    detail::local_device_barrier(access::fence_space::global_and_local);

#else
#ifdef _OPENMP
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
    
    for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
      dest[i * destStride] = src[i];
    detail::local_device_barrier(access::fence_space::global_and_local);

#else
#ifdef _OPENMP
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

  using host_barrier_type = std::function<void()>;

  group(id<Dimensions> group_id,
        range<Dimensions> local_range,
        range<Dimensions> num_groups,
        host_barrier_type* group_barrier = nullptr,
        id_type local_id = {},
        void *local_memory_ptr = nullptr)
  : _group_id{group_id}, 
    _local_range{local_range}, 
    _num_groups{num_groups},
    _group_barrier{group_barrier},
    _local_id{local_id},
    _local_memory_ptr(local_memory_ptr)
  {}

  HIPSYCL_KERNEL_TARGET
  void barrier() {
    (*_group_barrier)();
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const
  {
    return _local_id;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
    return _local_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const
  {
    return detail::linear_id<Dimensions>::get(_local_id,
                                              _local_range);
  }

  HIPSYCL_KERNEL_TARGET
  void *get_local_memory_ptr() const
  {
    return _local_memory_ptr;
  }

private:
  const id<Dimensions> _group_id;
  const range<Dimensions> _local_range;
  const range<Dimensions> _num_groups;
  const host_barrier_type* _group_barrier;
  const id_type _local_id;
  void *_local_memory_ptr;
public:
#endif

  HIPSYCL_KERNEL_TARGET
  bool leader() const {
    return get_local_linear_id() == 0;
  }

#ifdef SYCL_DEVICE_ONLY

  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_id() const
  {
    return detail::linear_id<Dimensions>::get(detail::get_local_id<Dimensions>(),
                                              detail::get_local_size<Dimensions>());
  }

  [[deprecated]]
  HIPSYCL_KERNEL_TARGET
  size_t get_linear_local_id() const
  {
    return get_local_linear_id();
  }

  template<typename workItemFunctionT>
  HIPSYCL_KERNEL_TARGET
  void parallelize_over_work_items(const range<1> flexibleRange,
                                  workItemFunctionT&& func) const
  {
    const range<1> physical_range = this->get_local_range();
    for(size_t i = __hipsycl_lid_x; i < flexibleRange.get(0); i += physical_range.get(0))
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
    for(size_t i = __hipsycl_lid_y; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = __hipsycl_lid_x; j < flexibleRange.get(1); j += physical_range.get(1))
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
    for(size_t i = __hipsycl_lid_z; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = __hipsycl_lid_y; j < flexibleRange.get(1); j += physical_range.get(1))
        for(size_t k = __hipsycl_lid_x; k < flexibleRange.get(2); k += physical_range.get(2))
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
    #ifdef _OPENMP
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
    #ifdef _OPENMP
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
  #ifdef _OPENMP
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
