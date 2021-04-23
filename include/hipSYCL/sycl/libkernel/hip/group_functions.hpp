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
T group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
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
inline bool group_any_of(sub_group g, bool pred) {
  return __any(pred);
}

// all_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(sub_group g, bool pred) {
  return __all(pred);
}

// none_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(sub_group g, bool pred) {
  return !__any(pred);
}

// reduce
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(sub_group g, T x, BinaryOperation binary_op) {
  auto     local_x = x;
  uint64_t activemask;
  asm("s_mov_b64 %0, exec" : "=r"(activemask));

  auto lid = g.get_local_linear_id();

  size_t lrange = g.get_local_range().size();

  group_barrier(g);

  for (size_t i = lrange / 2; i > 0; i /= 2) {
    auto other_x = detail::shuffle_impl(local_x, lid + i);

    // check if target thread exists/is active
    if (activemask & (1l << (lid + i)))
      local_x = binary_op(local_x, other_x);
  }
  return detail::shuffle_impl(local_x, 0);
}

// inclusive_scan
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(sub_group g, T x, BinaryOperation binary_op) {
  auto         local_x = x;
  const size_t lid     = g.get_local_linear_id();

  uint64_t activemask;
  asm("s_mov_b64 %0, exec" : "=r"(activemask));

  size_t lrange = g.get_local_linear_range();

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (activemask & (1l << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
}

// exclusive_scan
template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(sub_group g, V x, T init, BinaryOperation binary_op) {
  const size_t lid     = g.get_local_linear_id();
  auto         local_x = x;

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
