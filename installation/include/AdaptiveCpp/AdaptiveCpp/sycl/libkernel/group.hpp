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

#include "detail/thread_hierarchy.hpp"
#include "detail/device_barrier.hpp"

namespace hipsycl {
namespace sycl {

namespace vendor {
namespace hipsycl {
namespace synchronization {

struct none
{
  ACPP_KERNEL_TARGET
  static void run() {}
};

template<access::fence_space Fence_space>
struct barrier
{
  ACPP_KERNEL_TARGET
  static void run()
  {
    __acpp_if_target_device(
      detail::local_device_barrier(Fence_space);
    );
  }
};

using local_barrier = barrier<access::fence_space::local_space>;

template <
  access::fence_space Fence_space,
  access::mode Mode = access::mode::read_write
>
struct mem_fence
{
  ACPP_KERNEL_TARGET
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

#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  using host_barrier_type = std::function<void()>;
private:
  const id<Dimensions> _group_id;
  const range<Dimensions> _local_range;
  const range<Dimensions> _num_groups;
  // Don't store _group_barrier with type host_barrier_type*
  // to avoid function pointer types spilling into SSCP IR.
  const void* _group_barrier;
  const id_type _local_id;
  void *_local_memory_ptr;
public:

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

  ACPP_KERNEL_TARGET
  void *get_local_memory_ptr() const
  {
    return _local_memory_ptr;
  }
#endif

  ACPP_KERNEL_TARGET
  bool leader() const {
    return get_local_linear_id() == 0;
  }

  ACPP_KERNEL_TARGET
  id<Dimensions> get_group_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_group_id<Dimensions>(););
    return _group_id;
#endif
  }

  [[deprecated("To get the work group id use get_group_id() in SYCL 2020")]]
  ACPP_KERNEL_TARGET
  id<Dimensions> get_id() const
  {
    return get_group_id();
  }

  ACPP_KERNEL_TARGET
  size_t get_group_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_group_id<Dimensions>(dimension);)
    return _group_id[dimension];
#endif
  }

  [[deprecated("To get the work group id use get_group_id(int) in SYCL 2020")]]
  ACPP_KERNEL_TARGET
  size_t get_id(int dimension) const
  {
    return get_group_id(dimension);
  }

  [[deprecated("get_global_range() doesn't exist in SYCL 2020 anymore")]]
  ACPP_KERNEL_TARGET
  range<Dimensions> get_global_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>();
#else
    __acpp_if_target_sscp(return __acpp_sscp_get_global_size<Dimensions>(););
    return _num_groups * _local_range;
#endif
  }

  [[deprecated("get_global_range(int) doesn't exist in SYCL 2020 anymore")]]
  ACPP_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_global_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_global_size<Dimensions>(dimension););
    return _num_groups[dimension] * _local_range[dimension];
#endif
  }

  /// \return The physical local range for flexible work group sizes,
  /// the logical local range otherwise.
  ACPP_KERNEL_TARGET
  range<Dimensions> get_local_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(););
    return _local_range;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_size<Dimensions>(dimension););
    return _local_range[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_local_linear_range() const
  {
    __acpp_if_target_sscp(return __acpp_sscp_get_local_size<Dimensions>(););
    return get_local_range().size();
  }

  // Note: This returns the number of groups
  // in each dimension - earler versions of the spec wrongly 
  // claim that it should return the range "of the current group", 
  // i.e. the local range which makes no sense.
  ACPP_KERNEL_TARGET
  range<Dimensions> get_group_range() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(););
    return _num_groups;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_group_range(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_grid_size<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_grid_size<Dimensions>(dimension););
    return _num_groups[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_group_linear_range() const
  {
    __acpp_if_target_sscp(return __acpp_sscp_get_num_groups<Dimensions>(););
    return get_group_range().size();
  }

  ACPP_KERNEL_TARGET
  size_t operator[](int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_group_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_group_id<Dimensions>(dimension););
    return _group_id[dimension];
#endif
  }

  friend bool operator==(const group<Dimensions>& lhs, const group<Dimensions>& rhs){
    return lhs.get_group_id() == rhs.get_group_id() &&
           lhs.get_local_range() == rhs.get_local_range() &&
           lhs.get_group_range() == rhs.get_group_range();
  }

  friend bool operator!=(const group<Dimensions>& lhs, const group<Dimensions>& rhs){
    return !(lhs == rhs);
  }

  ACPP_KERNEL_TARGET
  size_t get_group_linear_id() const
  {
    __acpp_if_target_sscp(return __acpp_sscp_get_group_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(get_id(),
                                              get_group_range());
  }

  [[deprecated("Use get_group_linear_id() instead.")]]
  ACPP_KERNEL_TARGET
  size_t get_linear() const
  {
    return get_group_linear_id();
  }


  ACPP_KERNEL_TARGET
  id_type get_local_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>();
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(););
    return _local_id;
#endif
  }

  ACPP_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::get_local_id<Dimensions>(dimension);
#else
    __acpp_if_target_sscp(return detail::get_local_id<Dimensions>(dimension););
    return _local_id[dimension];
#endif
  }

  ACPP_KERNEL_TARGET
  linear_id_type get_local_linear_id() const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    return detail::linear_id<Dimensions>::get(detail::get_local_id<Dimensions>(),
                                              detail::get_local_size<Dimensions>());
#else
    __acpp_if_target_sscp(return __acpp_sscp_get_local_linear_id<Dimensions>(););
    return detail::linear_id<Dimensions>::get(_local_id,
                                              _local_range);
#endif
  }

  [[deprecated]]
  ACPP_KERNEL_TARGET
  size_t get_linear_local_id() const
  {
    return get_local_linear_id();
  }

  ACPP_KERNEL_TARGET
  void barrier() {
    __acpp_if_target_host(
      const host_barrier_type *barrier =
            static_cast<const host_barrier_type *>(_group_barrier);
      (*barrier)();
    );
    __acpp_if_target_device(
      detail::local_device_barrier();
    );
  }

  ACPP_KERNEL_TARGET
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
  ACPP_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
    __acpp_if_target_device(  
      h_item<Dimensions> idx{detail::get_local_id<Dimensions>(), detail::get_local_size<Dimensions>()};
      func(idx);
    );
#else
    __acpp_if_target_device(
      h_item<Dimensions> idx{detail::get_local_id<Dimensions>(), _local_range, _group_id, _num_groups};
      func(idx);
    );
#endif
    __acpp_if_target_host(
      iterate_over_work_items(_local_range, func);
    );
    Finalizer::run();
  }

  template<typename workItemFunctionT>
  ACPP_KERNEL_TARGET
  void parallel_for_work_item(workItemFunctionT func) const
  {
    parallel_for_work_item<vendor::hipsycl::synchronization::local_barrier>(func);
  }

  template<
    typename Finalizer,
    typename workItemFunctionT>
  ACPP_KERNEL_TARGET
  void parallel_for_work_item(range<Dimensions> flexibleRange,
                              workItemFunctionT func) const
  {
    __acpp_if_target_device(
      parallelize_over_work_items(flexibleRange, func);
    );
    __acpp_if_target_host(
      iterate_over_work_items(flexibleRange, func);
    );
    Finalizer::run();
  }

  template<typename workItemFunctionT>
  ACPP_KERNEL_TARGET
  void parallel_for_work_item(range<Dimensions> flexibleRange,
                              workItemFunctionT func) const
  {
    parallel_for_work_item<vendor::hipsycl::synchronization::local_barrier>(
      flexibleRange, func);
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
    // in hipSYCL, we do not need to distinguish between global and local pointers,
    // so we can just call the async_work_group_copy() variant that has
    // global_ptr as dest and local_ptr as source.
    
    global_ptr<dataT> global_dest{dest.get()};
    local_ptr<dataT> local_src{src.get()};
    
    return async_work_group_copy(global_dest, local_src, numElements);
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const
  {
    __acpp_if_target_device(
      const size_t physical_local_size = get_local_range().size();
      
      for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
        dest[i] = src[i];
      detail::local_device_barrier(access::fence_space::global_and_local);
    );
    __acpp_if_target_host(
      for(size_t i = 0; i < numElements; ++i)
        dest[i] = src[i];
    );

    return device_event{};
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const
  {
    __acpp_if_target_device(
      const size_t physical_local_size = get_local_range().size();
      
      for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
        dest[i] = src[i * srcStride];
      detail::local_device_barrier(access::fence_space::global_and_local);
    );
    __acpp_if_target_host(
      for(size_t i = 0; i < numElements; ++i)
        dest[i] = src[i * srcStride];
    );

    return device_event{};
  }

  template <typename dataT>
  ACPP_KERNEL_TARGET
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const
  {
    __acpp_if_target_device(
      const size_t physical_local_size = get_local_range().size();
      
      for(size_t i = get_local_linear_id(); i < numElements; i += physical_local_size)
        dest[i * destStride] = src[i];
      detail::local_device_barrier(access::fence_space::global_and_local);
    );
    __acpp_if_target_host(
      for(size_t i = 0; i < numElements; ++i)
        dest[i * destStride] = src[i];
    );

    return device_event{};
  }

  template <typename... eventTN>
  ACPP_KERNEL_TARGET
  void wait_for(eventTN...) const {}

private:

  // The parallelize_over_* functions assume that the code is executed
  // a number of times in parallel equal to the physical group size.
  // This is not supported on host.
  template<typename workItemFunctionT>
  ACPP_KERNEL_TARGET
  void parallelize_over_work_items(const range<1> flexibleRange,
                                  workItemFunctionT&& func) const
  {
    const range<1> physical_range = this->get_local_range();
    for(size_t i = __acpp_lid_x; i < flexibleRange.get(0); i += physical_range.get(0))
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
  ACPP_KERNEL_TARGET
  void parallelize_over_work_items(const range<2> flexibleRange,
                                  workItemFunctionT&& func) const
  {
    const range<2> physical_range = this->get_local_range();
    // Reverse dimensions of hipThreadIdx_* compared to flexibleRange.get()
    // to make sure that the fastest index in SYCL terminology is mapped
    // to the fastest index of the backend
    for(size_t i = __acpp_lid_y; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = __acpp_lid_x; j < flexibleRange.get(1); j += physical_range.get(1))
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
  ACPP_KERNEL_TARGET
  void parallelize_over_work_items(const range<3> flexibleRange,
                                  workItemFunctionT&& func) const
  { 
    const range<3> physical_range = this->get_local_range();
    for(size_t i = __acpp_lid_z; i < flexibleRange.get(0); i += physical_range.get(0))
      for(size_t j = __acpp_lid_y; j < flexibleRange.get(1); j += physical_range.get(1))
        for(size_t k = __acpp_lid_x; k < flexibleRange.get(2); k += physical_range.get(2))
        {
  #ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          h_item<3> idx{id<3>{i,j,k}, flexibleRange};
  #else
          h_item<3> idx{id<3>{i,j,k}, flexibleRange, _group_id, _num_groups};
  #endif
          func(idx);
        }
  }

  /// The iteratate_over functions are intended for use on host,
  /// without support for ondemand iteration space info
  /// Ondemand iteration space info is unsupported.
  template<typename workItemFunctionT>
  void iterate_over_work_items(const range<1> iteration_range,
                              workItemFunctionT&& func) const
  {
#ifdef _OPENMP
    #pragma omp simd 
#endif
    for(size_t i = 0; i < iteration_range.get(0); ++i)
    {
#ifndef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
      h_item<1> idx{
        id<1>{i}, 
        iteration_range, _group_id, _num_groups
      };
      func(idx);
#endif
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
#ifndef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
        h_item<2> idx{
          id<2>{i,j}, 
          iteration_range, _group_id, _num_groups
        };

        func(idx);
#endif
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
#ifndef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
          h_item<3> idx{
            id<3>{i,j,k}, 
            iteration_range, _group_id, _num_groups
          };

          func(idx);
#endif
        }
    // No memfence is needed here, because on CPU we only have one physical thread per work group.
  }

};

}
}

#endif
