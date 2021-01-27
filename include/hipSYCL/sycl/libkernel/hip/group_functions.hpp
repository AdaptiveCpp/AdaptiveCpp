/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifdef SYCL_DEVICE_ONLY
#ifdef HIPSYCL_PLATFORM_HIP

#ifndef HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../generic/hiplike/warp_shuffle.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

// broadcast
template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x, typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::shuffle_impl(x, static_cast<int>(local_linear_id));
}

// barrier
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(Group g, memory_scope fence_scope = Group::fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  }
  __syncthreads();
}

template<>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(sub_group g, memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
  // threads run in lock-step no sync needed
}

// any_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(sub_group g, bool pred) { return __any(pred); }

// all_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(sub_group g, bool pred) { return __all(pred); }

// none_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(sub_group g, bool pred) { return !__any(pred); }

// reduce
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(sub_group g, T x, BinaryOperation binary_op) {
  auto     local_x = x;
  uint64_t activemask;
  asm("s_mov_b64 %0, exec" : "=r"(activemask));

  // adaption of rocprim dpp_reduce
  // quad_perm: add 0+1, 2+3
  local_x = binary_op(detail::warp_move_dpp<T, 0xb1>(local_x), local_x);
  // quad_perm: add 0+2
  local_x = binary_op(detail::warp_move_dpp<T, 0x4e>(local_x), local_x);
  // row_sr: add 0+4
  local_x = binary_op(detail::warp_move_dpp<T, 0x114>(local_x), local_x);
  // row_sr: add 0+8
  local_x = binary_op(detail::warp_move_dpp<T, 0x118>(local_x), local_x);
  // row_bcast15: add 0+15
  local_x = binary_op(detail::warp_move_dpp<T, 0x142>(local_x), local_x);

  if constexpr (warpSize > 32) {
    // row_bcast31: add 0+31
    local_x = binary_op(detail::warp_move_dpp<T, 0x143>(local_x), local_x);
  }

  // get the result from last thead
  return detail::shuffle_impl(local_x, warpSize - 1);
}

// inclusive_scan
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(sub_group g, T x, BinaryOperation binary_op) {
  auto local_x = x;
  auto lid     = g.get_local_linear_id();

  uint64_t activemask;
  asm("s_mov_b64 %0, exec" : "=r"(activemask));

  auto row_id  = lid % 16;
  auto lane_id = lid % warpSize;
  // adaption of rocprim dpp_scan
  T tmp;
  // row_sr:1
  tmp = binary_op(detail::warp_move_dpp<T, 0x111>(local_x), local_x);
  if (row_id > 0)
    local_x = tmp;

  // row_sr:2
  tmp = binary_op(detail::warp_move_dpp<T, 0x112>(local_x), local_x);
  if (row_id > 1)
    local_x = tmp;

  // row_sr:4
  tmp = binary_op(detail::warp_move_dpp<T, 0x114>(local_x), local_x);
  if (row_id > 3)
    local_x = tmp;

  // row_sr:8
  tmp = binary_op(detail::warp_move_dpp<T, 0x118>(local_x), local_x);
  if (row_id > 7)
    local_x = tmp;

  // row_bcast15
  tmp = binary_op(detail::warp_move_dpp<T, 0x142>(local_x), local_x);
  if (lane_id % 32 > 15)
    local_x = tmp;

  if constexpr (warpSize > 32) {
    // row_bcast31
    tmp = binary_op(detail::warp_move_dpp<T, 0x143>(local_x), local_x);
    if (lane_id > 31)
      local_x = tmp;
  }

  return local_x;
}

// exclusive_scan
template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(sub_group g, V x, T init, BinaryOperation binary_op) {
  auto lid     = g.get_local_linear_id();
  auto local_x = x;

  local_x = detail::shuffle_up_impl(local_x, 1);
  if (lid == 0)
    local_x = init;

  return group_inclusive_scan(g, local_x, binary_op);
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

#endif // HIPSYCL_PLATFORM_HIP
#endif // SYCL_DEVICE_ONLY
