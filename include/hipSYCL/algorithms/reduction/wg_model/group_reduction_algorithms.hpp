
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

#include "hipSYCL/sycl/libkernel/detail/local_memory_allocator.hpp"
#include "hipSYCL/sycl/libkernel/group_functions.hpp"

#include "hipSYCL/sycl/libkernel/sp_group.hpp"
#include "wg_model_queries.hpp"
#include "../reduction_descriptor.hpp"

#ifndef HIPSYCL_REDUCTION_GROUP_REDUCTION_HPP
#define HIPSYCL_REDUCTION_GROUP_REDUCTION_HPP

namespace hipsycl::algorithms::reduction::wg_model::group_reductions {

template<int Dim>
void local_barrier(sycl::nd_item<Dim> idx) {
  sycl::group_barrier(idx.get_group());
}

template<int Dim>
void local_barrier(sycl::group<Dim> idx) {
  sycl::group_barrier(idx);
}

template<class T>
void local_barrier(const T&) {
  sycl::detail::local_device_barrier(sycl::access::fence_space::local_space);
}

template<typename... T>
class local_memory_request_bundle {
  template <class U>
  static void extract_max_align_and_size(U, std::size_t &alignment_out,
                                         std::size_t &size_out) {
    if(alignof(U) > alignment_out) {
      alignment_out = alignof(U);
    }
    if(sizeof(U) > size_out) {
      size_out = sizeof(U);
    }
  }

public:
  local_memory_request_bundle(std::size_t& currently_allocated_local_mem_size,
                              std::size_t num_elements) {
    std::size_t max_align = 0;
    std::size_t max_elem_size = 0;
    const std::size_t min_alignment = 4;

    (extract_max_align_and_size(T{}, max_align, max_elem_size), ...);

    max_align = std::max(max_align, min_alignment);

    sycl::detail::local_memory_allocator alloc{currently_allocated_local_mem_size};
    _addr = alloc.alloc(max_align, max_elem_size * num_elements);
    currently_allocated_local_mem_size = alloc.get_allocation_size();
  }

  // Only available in device code
  void* get_device_address() const {
    return get_device_address(_addr);
  }

  static void* get_device_address(sycl::detail::local_memory_allocator::address addr) {
    return sycl::detail::local_memory::get_ptr<void>(addr);
  }

  sycl::detail::local_memory_allocator::address get_address() const {
    return _addr;
  }
private:
  sycl::detail::local_memory_allocator::address _addr;
};

// Only supported in modes where work groups exist!
// Therefore, not supported on host basic parallel_for
template<typename... ReductionDescriptors>
class generic_local_memory {
private:

  void initialize(std::size_t& allocated_local_mem) {
    local_memory_request_bundle<typename ReductionDescriptors::value_type...> request{
        allocated_local_mem, _group_size};
    _addr = request.get_address();

    bool all_reductions_have_known_identity =
        (ReductionDescriptors::has_known_identity() && ...);
    if(!all_reductions_have_known_identity) {
      // We need additional scratch to store initialization state
      local_memory_request_bundle<initialization_flag_t> init_state_request{
        allocated_local_mem, _group_size};
      _init_state_addr = init_state_request.get_address();
    }
  }
public:
  generic_local_memory() = default;
  generic_local_memory(std::size_t& currently_allocated_local_mem_size, std::size_t group_size)
  : _group_size{group_size} {
    initialize(currently_allocated_local_mem_size);
  }

  template <class ReductionStage>
  void configure_for_stage(ReductionStage &s, std::size_t stage_index,
                           std::size_t num_stages,
                           const ReductionDescriptors &...all_reductions) {
    // Configuration for the first stage will already implicitly happen in the
    // constructor generic_local_memory(std::size_t& currently_allocated_local_mem_size,
    // std::size_t group_size).
    // First stage needs to be treated differently, because it might also use user-controlled
    // local memory for the user code contained in the kernel.
    if(stage_index != 0) {
      std::size_t allocated_local_mem = 0;
      initialize(allocated_local_mem);
      s.local_mem = allocated_local_mem;
    }
  }

  template <class WiIndex, class WiReducer, class ConfiguredReductionDescriptor>
  auto operator()(const WiIndex &wi,
                  const ConfiguredReductionDescriptor &descriptor,
                  const WiReducer& wi_reducer,
                  bool &is_leader, bool &result_is_initialized) const {
    using value_type = typename ConfiguredReductionDescriptor::value_type;

    std::size_t my_lid = get_local_linear_id(wi);
    is_leader = (my_lid == 0);

    value_type *local_memory = static_cast<value_type *>(
        local_memory_request_bundle<>::get_device_address(_addr));

    if constexpr(ConfiguredReductionDescriptor::has_known_identity()) {
      result_is_initialized = true;

      local_memory[my_lid] = wi_reducer.value();
      local_barrier(wi);

      const int local_size = _group_size;
      for (int i = local_size / 2; i > 0; i /= 2) {
        if(my_lid < i)
          local_memory[my_lid] =
              descriptor.get_operator()(local_memory[my_lid], local_memory[my_lid + i]);
        local_barrier(wi);
      }
      return local_memory[0];

    } else {

      initialization_flag_t *initialization_state_local_memory =
          static_cast<initialization_flag_t *>(
              local_memory_request_bundle<>::get_device_address(
                  _init_state_addr));

      initialization_state_local_memory[my_lid] = wi_reducer.is_initialized();
      if(wi_reducer.is_initialized()) {
        local_memory[my_lid] = wi_reducer.value();
      }

      local_barrier(wi);

      const int local_size = _group_size;
      for (int i = local_size / 2; i > 0; i /= 2) {
        if(my_lid < i) {
          initialization_flag_t is_lhs_initialized = initialization_state_local_memory[my_lid    ];
          initialization_flag_t is_rhs_initialized = initialization_state_local_memory[my_lid + i];
          if(is_lhs_initialized && is_rhs_initialized) {
            local_memory[my_lid] =
              descriptor.get_operator()(local_memory[my_lid], local_memory[my_lid + i]);
            initialization_state_local_memory[my_lid] = true;
          } else if(is_rhs_initialized) { // In this branch, is_lhs_initialized must be false
            local_memory[my_lid] = local_memory[my_lid + i];
            initialization_state_local_memory[my_lid] = true;
          }
        }
        local_barrier(wi);
      }
      result_is_initialized = initialization_state_local_memory[0];
      if(result_is_initialized)
        return local_memory[0];
      else
        return value_type{};
    }
    
  }

private:
  sycl::detail::local_memory_allocator::address _addr;
  sycl::detail::local_memory_allocator::address _init_state_addr;
  std::size_t _group_size;
};

} // namespace hipsycl::algorithms::reduction::wg_model::group_reductions

#endif
