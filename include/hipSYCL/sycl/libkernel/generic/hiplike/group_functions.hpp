/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#include "../../backend.hpp"
#include "../../detail/data_layout.hpp"
#include "../../detail/thread_hierarchy.hpp"
#include "warp_shuffle.hpp"
#include "../../id.hpp"
#include "../../sub_group.hpp"
#include "../../vec.hpp"
#include "warp_shuffle.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename T, int N>
void writeToMemory(T *scratch, vec<T, N> v) {
  if constexpr (1 <= N)
    scratch[0] = v.s0();
  if constexpr (2 <= N)
    scratch[1] = v.s1();
  if constexpr (3 <= N)
    scratch[2] = v.s2();
  if constexpr (4 <= N)
    scratch[3] = v.s3();
  if constexpr (8 <= N) {
    scratch[4] = v.s4();
    scratch[5] = v.s5();
    scratch[6] = v.s6();
    scratch[7] = v.s7();
  }
  if constexpr (16 <= N) {
    scratch[8]  = v.s8();
    scratch[9]  = v.s9();
    scratch[10] = v.sA();
    scratch[11] = v.sB();
    scratch[12] = v.sC();
    scratch[13] = v.sD();
    scratch[14] = v.sE();
    scratch[15] = v.sF();
  }
}

template<typename T, int N>
void readFromMemory(T *scratch, vec<T, N> &v) {
  if constexpr (1 <= N)
    v.s0() = scratch[0];
  if constexpr (2 <= N)
    v.s1() = scratch[1];
  if constexpr (3 <= N)
    v.s2() = scratch[2];
  if constexpr (4 <= N)
    v.s3() = scratch[3];
  if constexpr (8 <= N) {
    v.s4() = scratch[4];
    v.s5() = scratch[5];
    v.s6() = scratch[6];
    v.s7() = scratch[7];
  }
  if constexpr (16 <= N) {
    v.s8() = scratch[8];
    v.s9() = scratch[9];
    v.sA() = scratch[10];
    v.sB() = scratch[11];
    v.sC() = scratch[12];
    v.sD() = scratch[13];
    v.sE() = scratch[14];
    v.sF() = scratch[15];
  }
}

// reduce implementation
template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op, T *scratch) {

  auto   lid    = g.get_local_linear_id();
  size_t lrange = g.get_local_range().size();

  if (lrange == 1)
    return x;

  scratch[lid] = x;
  group_barrier(g);

  size_t outputs = lrange;
  for (size_t i = (lrange + 1) / 2; i > 1; i = (i + 1) / 2) {
    if (lid < i && lid + i < outputs)
      scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
    outputs = outputs / 2 + outputs % 2;
    group_barrier(g);
  }

  return binary_op(scratch[0], scratch[1]);
}

template<typename Group, typename T, int N, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
vec<T, N> group_reduce(Group g, vec<T, N> x, BinaryOperation binary_op, T *scratch) {
  auto   lid    = g.get_local_linear_id();
  size_t lrange = g.get_local_range().size();

  if (lrange == 1)
    return x;

  detail::writeToMemory(scratch + lid * N, x);
  group_barrier(g);

  size_t outputs = lrange;
  for (size_t i = (lrange + 1) / 2; i > 1; i = (i + 1) / 2) {

    if (lid < i && lid + i < outputs) {
      vec<T, N> v1, v2;

      detail::readFromMemory(scratch + lid * N, v1);
      detail::readFromMemory(scratch + (lid + i) * N, v2);

      detail::writeToMemory(scratch + lid * N, binary_op(v1, v2));
    }

    outputs = outputs / 2 + outputs % 2;
    group_barrier(g);
  }

  vec<T, N> v1, v2;

  detail::readFromMemory(scratch, v1);
  detail::readFromMemory(scratch + N, v2);
  return binary_op(v1, v2);
}

// any_of
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, T *first, T *last) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= *p;

  return group_any_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= pred(*p);

  return group_any_of(g, local);
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, T *first, T *last) {
  return any_of(g, first, last);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, T *first, T *last, Predicate pred) {
  return any_of(g, first, last, pred);
}

// all_of
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, T *first, T *last) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local &= *p;

  return group_all_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local &= pred(*p);

  return group_all_of(g, local);
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, T *first, T *last) {
  return all_of(g, first, last);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, T *first, T *last, Predicate pred) {
  return all_of(g, first, last, pred);
}

// none_of
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, T *first, T *last) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = *start_ptr;

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= *p;

  return group_none_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, T *first, T *last, Predicate pred) {
  auto group_range        = g.get_local_range().size();
  auto elements_per_group = (last - first + group_range - 1) / group_range;
  T *  start_ptr          = first + elements_per_group * g.get_local_linear_id();
  T *  end_prt            = start_ptr + elements_per_group;

  if (end_prt > last)
    end_prt = last;

  auto local = pred(*start_ptr);

  for (T *p = start_ptr + 1; p < end_prt; ++p)
    local |= pred(*p);

  return group_none_of(g, local);
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, T *first, T *last) {
  return none_of(g, first, last);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, T *first, T *last, Predicate pred) {
  return none_of(g, first, last, pred);
}


// reduce
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  __shared__ std::aligned_storage_t<sizeof(T) * 1024, alignof(T)> scratch_storage;
  T *                                                             scratch = reinterpret_cast<T *>(&scratch_storage);

  const size_t group_range         = g.get_local_range().size();
  const size_t num_elements        = last - first;
  const size_t lid                 = g.get_local_linear_id();

  T *start_ptr = first + lid;

  if (num_elements == 1)
    return *first;

  if (num_elements >= group_range) {
    auto local = *start_ptr;

    for (T *p = start_ptr + group_range; p < last; p += group_range)
      local = binary_op(local, *p);

    return detail::group_reduce(g, local, binary_op, scratch);
  } else {
    const size_t warp_id           = lid / warpSize;
    const size_t num_warps         = num_elements / warpSize;
    const size_t elements_per_warp = (num_elements + num_warps - 1) / num_warps;
    size_t       outputs           = num_elements;

    if (num_warps == 0 && lid < num_elements) {
      scratch[lid] = first[lid];
    } else if (warp_id < num_warps) {
      const size_t active_threads = num_warps * warpSize;

      auto local = *start_ptr;

      for (T *p = start_ptr + active_threads; p < last; p += active_threads)
        local = binary_op(local, *p);

      sub_group sg{};

      local = group_reduce(sg, local, binary_op);
      if (sg.leader())
        scratch[warp_id] = local;
      outputs = num_warps;
    }
    group_barrier(g);

    for (size_t i = (outputs + 1) / 2; i > 1; i = (i + 1) / 2) {
      if (lid < i && lid + i < outputs)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      outputs = outputs / 2 + outputs % 2;
      group_barrier(g);
    }

    return binary_op(scratch[0], scratch[1]);
  }
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, V *first, V *last, T init, BinaryOperation binary_op) {
  return binary_op(reduce(g, first, last, binary_op), init);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, V *first, V *last, T init, BinaryOperation binary_op) {
  return reduce(g, first, last, init, binary_op);
}

template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, BinaryOperation binary_op) { return reduce(g, first, last, binary_op); }

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, T init, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();

  if (g.leader()) {
    if (first == last)
      return result;

    *(result++) = binary_op(init, *(first++));
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return group_broadcast(g, result);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();

  if (g.leader()) {
    if (first == last)
      return result;

    *(result++) = *(first++);
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return group_broadcast(g, result);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result, T init, BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, binary_op);
}


// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, T init, BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();
  if (lid == 0 && last - first > 0)
    result[0] = init;
  return inclusive_scan(g, first, last - 1, result + 1, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, T{}, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result, T init, BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *eleader_xclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, binary_op);
}

} // namespace detail

// broadcast
template<typename Group, typename T, typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  __shared__ T scratch[1];
  auto         lid = g.get_local_linear_id();

  if (lid == local_linear_id)
    scratch[0] = x;
  group_barrier(g);

  return scratch[0];
}

template<typename Group, typename T, int N, typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
vec<T, N> group_broadcast(Group g, vec<T, N> x, typename Group::linear_id_type local_linear_id = 0) {
  __shared__ T scratch[N];
  auto         lid = g.get_local_linear_id();

  if (lid == local_linear_id)
    detail::writeToMemory(scratch, x);
  group_barrier(g);

  detail::readFromMemory(scratch, x);

  return x;
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  auto target_lid = detail::linear_id<g.dimensions>::get(local_id, detail::get_local_size<g.dimensions>());

  return group_broadcast(g, x, target_lid);
}

// any_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(Group g, bool pred) { return __syncthreads_or(pred); }

// all_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(Group g, bool pred) { return __syncthreads_and(pred); }

// none_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(Group g, bool pred) { return !__syncthreads_or(pred); }

// reduce
template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  __shared__ T scratch[1024];

  return detail::group_reduce(g, x, binary_op, scratch);
}

template<typename Group, typename T, int N, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
vec<T, N> group_reduce(Group g, vec<T, N> x, BinaryOperation binary_op) {
  __shared__ T scratch[1024 * N];

  return detail::group_reduce(g, x, binary_op, scratch);
}

// exclusive_scan
template<typename Group, typename T, int N, typename V, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
vec<T, N> group_exclusive_scan(Group g, vec<T, N> x, V init, BinaryOperation binary_op) {
  __shared__ T scratch[1024 * N];
  auto         lid               = g.get_local_linear_id();
  auto         group_local_range = g.get_local_range().size();

  if (lid + 1 < group_local_range) {
    detail::writeToMemory(scratch + (lid + 1) * N, x);
  } else {
    detail::writeToMemory(scratch, init);
  }
  group_barrier(g);

  for (size_t i = 1; i < group_local_range; i *= 2) {
    size_t    next_id = lid - i;
    vec<T, N> v1, v2;
    if (i <= lid && lid < group_local_range) {
      detail::readFromMemory(scratch + lid * N, v1);
      detail::readFromMemory(scratch + next_id * N, v2);
    }
    group_barrier(g);
    if (i <= lid && lid < group_local_range) {
      detail::writeToMemory(scratch + lid * N, binary_op(v1, v2));
    }
    group_barrier(g);
  }

  vec<T, N> v;
  detail::readFromMemory(scratch + lid * N, v);
  return v;
}

template<typename Group, typename T, typename V, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, T x, V init, BinaryOperation binary_op) {
  __shared__ T scratch[1024];
  auto         lid               = g.get_local_linear_id();
  auto         group_local_range = g.get_local_range().size();

  if (lid + 1 < group_local_range) {
    scratch[lid + 1] = x;
  } else {
    scratch[0] = init;
  }
  group_barrier(g);

  for (size_t i = 1; i < group_local_range; i *= 2) {
    size_t next_id = lid - i;
    T      local_x;
    T      other_x;
    if (i <= lid && lid < group_local_range) {
      local_x = scratch[lid];
      other_x = scratch[next_id];
    }

    group_barrier(g);
    if (i <= lid && lid < group_local_range)
      scratch[lid] = binary_op(local_x, other_x);
    group_barrier(g);
  }

  return scratch[lid];
}

// inclusive_scan
template<typename Group, typename T, int N, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
vec<T, N> group_inclusive_scan(Group g, vec<T, N> x, BinaryOperation binary_op) {
  __shared__ T scratch[1024 * N];
  auto         lid               = g.get_local_linear_id();
  auto         group_local_range = g.get_local_range().size();

  detail::writeToMemory(scratch + lid * N, x);
  group_barrier(g);

  for (size_t i = 1; i < group_local_range; i *= 2) {
    size_t    next_id = lid - i;
    vec<T, N> v1, v2;
    if (i <= lid && lid < group_local_range) {
      detail::readFromMemory(scratch + lid * N, v1);
      detail::readFromMemory(scratch + next_id * N, v2);
    }
    group_barrier(g);
    if (i <= lid && lid < group_local_range) {
      detail::writeToMemory(scratch + lid * N, binary_op(v1, v2));
    }
    group_barrier(g);
  }

  vec<T, N> v;
  detail::readFromMemory(scratch + lid * N, v);
  return v;
}

template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  __shared__ T scratch[1024];
  auto         lid               = g.get_local_linear_id();
  auto         group_local_range = g.get_local_range().size();

  scratch[lid] = x;
  group_barrier(g);

  for (size_t i = 1; i < group_local_range; i *= 2) {
    size_t next_id = lid - i;
    T      local_x;
    T      other_x;
    if (i <= lid && lid < group_local_range) {
      local_x = scratch[lid];
      other_x = scratch[next_id];
    }

    group_barrier(g);
    if (i <= lid && lid < group_local_range)
      scratch[lid] = binary_op(local_x, other_x);
    group_barrier(g);
  }

  return scratch[lid];
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
