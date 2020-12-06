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
#ifdef HIPSYCL_PLATFORM_CUDA

#ifndef HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

constexpr unsigned int AllMask = 0xFFFFFFFF;

template<typename T,
         typename std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
__device__
T shuffle_impl(T x, int id) {
  return __shfl_sync(detail::AllMask, x, id);
}

template<>
__device__
char shuffle_impl(char x, int id) {
  return static_cast<char>(shuffle_impl(static_cast<int>(x), id));
}

template<typename T, int N>
__device__
sycl::vec<T, N> shuffle_impl(sycl::vec<T, N> x, int id) {
  sycl::vec<T, N> ret{};

  if constexpr (1 <= N)
    ret.s0() = shuffle_impl(x.s0(), id);
  if constexpr (2 <= N)
    ret.s1() = shuffle_impl(x.s1(), id);
  if constexpr (3 <= N)
    ret.s2() = shuffle_impl(x.s2(), id);
  if constexpr (4 <= N)
    ret.s3() = shuffle_impl(x.s3(), id);
  if constexpr (8 <= N) {
    ret.s4() = shuffle_impl(x.s4(), id);
    ret.s5() = shuffle_impl(x.s5(), id);
    ret.s6() = shuffle_impl(x.s6(), id);
    ret.s7() = shuffle_impl(x.s7(), id);
  }
  if constexpr (16 <= N) {
    ret.s8() = shuffle_impl(x.s8(), id);
    ret.s9() = shuffle_impl(x.s9(), id);
    ret.sA() = shuffle_impl(x.sA(), id);
    ret.sB() = shuffle_impl(x.sB(), id);
    ret.sC() = shuffle_impl(x.sC(), id);
    ret.sD() = shuffle_impl(x.sD(), id);
    ret.sE() = shuffle_impl(x.sE(), id);
    ret.sF() = shuffle_impl(x.sF(), id);
  }

  return ret;
}

} // namespace detail

// broadcast
template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::shuffle_impl(x, local_linear_id);
}

template<typename T, int N>
HIPSYCL_KERNEL_TARGET
sycl::vec<T, N> group_broadcast(sub_group g, sycl::vec<T, N> x,
                                typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::shuffle_impl(x, local_linear_id);
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
  __syncwarp(); // not necessarily needed, but might improve performance
}


// any_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(sub_group g, bool pred) {
  return __any_sync(detail::AllMask, pred);
}


// all_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(sub_group g, bool pred) {
  return __all_sync(detail::AllMask, pred);
}


// none_of
template<>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(sub_group g, bool pred) {
  return !__any_sync(detail::AllMask, pred);
}


// reduce
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(sub_group g, T x, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();

  size_t lrange          = 1;
  auto group_local_range = g.get_local_range();
  for (int i = 0; i < g.dimensions; ++i)
    lrange *= group_local_range[i];

  group_barrier(g);

  auto local_x = x;

  for (size_t i = lrange / 2; i > 0; i /= 2) {
    auto other_x = detail::shuffle_impl(local_x, lid + i);
    if (lid < i)
      local_x = binary_op(local_x, other_x);
  }
  return detail::shuffle_impl(local_x, 0);
}


// exclusive_scan
template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(sub_group g, V x, T init, BinaryOperation binary_op) {
  auto lid      = g.get_local_linear_id();
  size_t lrange = g.get_local_linear_range();

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  size_t next_id = lid - 1;
  if (g.leader())
    next_id = 0;

  auto return_value = detail::shuffle_impl(local_x, lid - 1);

  if (g.leader())
    return init;

  return binary_op(return_value, init);
}


// inclusive_scan
template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(sub_group g, T x, BinaryOperation binary_op) {
  auto lid      = g.get_local_linear_id();
  size_t lrange = g.get_local_linear_range();

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
}


} // namespace sycl
} // namespace hipsycl

#endif //HIPSYCL_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP

#endif //HIPSYCL_PLATFORM_CUDA
#endif //SYCL_DEVICE_ONLY
