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

#ifndef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"

#include "rv_shuffle.hpp"

namespace hipsycl {
namespace sycl {

template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(sub_group g, T x, BinaryOperation binary_op) {
#ifdef HIPSYCL_HAS_RV
  const size_t       lid        = g.get_local_linear_id();
  const unsigned int activemask = rv_mask();

  auto local_x = x;

  for (size_t i = rv_num_lanes() / 2; i > 0; i /= 2) {
    auto other_x = detail::shuffle_down_impl(local_x, i);
    if (activemask & (1 << (lid + i)))
      local_x = binary_op(local_x, other_x);
  }
  return detail::extract_impl(local_x, 0);
#else
  return x;
#endif
}

namespace detail {
// reduce implementation
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op, T *scratch) {
#ifdef HIPSYCL_HAS_RV
  const auto   lid    = g.get_local_linear_id();
  const std::size_t local_range = g.get_local_range().size();
  sub_group    sg{};

  scratch[lid] = x;
  group_barrier(g);

  size_t i = 1;

  if(lid < rv_num_lanes() && rv_num_lanes() <= local_range) {
    x = group_reduce(sg, scratch[rv_lane_id()], binary_op);
    for(i = rv_num_lanes(); i + rv_num_lanes() <= local_range; i += rv_num_lanes()) {
      x = binary_op(x, group_reduce(sg, scratch[i + rv_lane_id()], binary_op));
    }
  }
  if(g.leader()){
    for(; i < local_range; ++i)
      x = binary_op(x, scratch[i]);
    scratch[0] = x;
  }

  group_barrier(g);
  x = scratch[0];
  group_barrier(g);

  return x;
#else
  const size_t lid = g.get_local_linear_id();

  scratch[lid] = x;
  group_barrier(g);

  if (g.leader()) {
    T result = scratch[0];

    for (int i = 1; i < g.get_local_range().size(); ++i)
      result = binary_op(result, scratch[i]);

    scratch[0] = result;
  }

  group_barrier(g);
  T tmp = scratch[0];
  group_barrier(g);

  return tmp;
#endif
}

// functions using pointers
// any_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, Ptr first, Ptr last) {
  bool result = false;

  if (g.leader()) {
    while (first < last) {
      if (*(first++)) {
        result = true;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = false;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = true;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, Ptr first, Ptr last) {
  const bool result = leader_any_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = leader_any_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// all_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, Ptr first, Ptr last) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (!*(first++)) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (!pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, Ptr first, Ptr last) {
  const bool result = leader_all_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const bool result = leader_all_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// none_of
template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, Ptr first, Ptr last) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (*(first++)) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool leader_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    while (first != last) {
      if (pred(*(first++))) {
        result = false;
        break;
      }
    }
  }
  return result;
}

template<typename Group, typename Ptr>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, Ptr first, Ptr last) {
  auto result = leader_none_of(g, first, last);
  return group_broadcast(g, result);
}

template<typename Group, typename Ptr, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  auto result = leader_none_of(g, first, last, pred);
  return group_broadcast(g, result);
}

// reduce
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  T result{};

  if (first >= last) {
    return T{};
  }

  if (g.leader()) {
#pragma omp simd
    for (T *i = first; i < last; ++i)
      result = binary_op(result, *i);
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, V init, BinaryOperation binary_op) {
  auto result = leader_reduce(g, first, last, binary_op);

  if (g.leader()) {
    result = binary_op(result, init);
  }
  return result;
}

template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  T result{};

  if (first >= last) {
    return T{};
  }

  if (g.leader()) {
    result = *(first++);
    while (first != last)
      result = binary_op(result, *(first++));
  }
  return group_broadcast(g, result);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce(Group g, V *first, V *last, T init, BinaryOperation binary_op) {
  const auto result = leader_reduce(g, first, last, init, binary_op);

  return group_broadcast(g, result);
}

// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {

  if (g.leader()) {
    *(result++) = init;
    while (first != last - 1) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_exclusive_scan(g, first, last, result, T{}, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
  const auto ret = leader_exclusive_scan(g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, T{}, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {
  if (first == last)
    return result;

  if (g.leader()) {
    *(result++) = binary_op(init, *(first++));
    while (first != last) {
      *result = binary_op(*(result - 1), *(first++));
      result++;
    }
  }
  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_inclusive_scan(g, first, last, result, T{}, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
  auto ret = leader_inclusive_scan(g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, T{}, binary_op);
}

} // namespace detail

// broadcast
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  if (lid == local_linear_id) {
    scratch[0] = x;
  }

  group_barrier(g);
  T tmp = scratch[0];
  group_barrier(g);

  return tmp;
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  const size_t target_lid =
      detail::linear_id<g.dimensions>::get(local_id, g.get_local_range());
  return group_broadcast(g, x, target_lid);
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
#ifdef HIPSYCL_HAS_RV
  return detail::extract_impl(x, local_linear_id);
#else
  return x;
#endif
}

// barrier
template<typename Group>
HIPSYCL_KERNEL_TARGET [[clang::annotate("hipsycl_splitter")]]
inline void group_barrier(Group g, memory_scope fence_scope = Group::fence_scope) {
  if (fence_scope == memory_scope::work_item) {
    // doesn't need sync
  } else if (fence_scope == memory_scope::sub_group) {
    // doesn't need sync (sub_group size = 1 or vectorization front)
  } else if (fence_scope == memory_scope::work_group) {
    g.barrier();
  } else if (fence_scope == memory_scope::device) {
    g.barrier();
  }
}

template<>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(sub_group g, memory_scope fence_scope) {
  // doesn't need sync
}

// any_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = false;
  group_barrier(g);

  if (pred)
    scratch[0] = pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(sub_group, bool pred) {
#ifdef HIPSYCL_HAS_RV
  return rv_any(pred);
#else
  return pred;
#endif
}

// all_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = true;
  group_barrier(g);

  if (!pred)
    scratch[0] = pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(sub_group, bool pred) {
#ifdef HIPSYCL_HAS_RV
  return rv_all(pred);
#else
  return pred;
#endif
}

// none_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(Group g, bool pred) {
  bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

  scratch[0] = true;
  group_barrier(g);

  if (pred)
    scratch[0] = !pred;

  group_barrier(g);

  bool tmp = scratch[0];

  group_barrier(g);

  return tmp;
}

template<>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(sub_group, bool pred) {
#ifdef HIPSYCL_HAS_RV
  return rv_all(!pred);
#else
  return pred;
#endif
}

// reduce
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  T tmp = detail::group_reduce(g, x, binary_op, scratch);

  return tmp;
}

template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
  T group_exclusive_scan(sub_group g, V x, T init, BinaryOperation binary_op) {
#ifdef HIPSYCL_HAS_RV
  const size_t       lid        = g.get_local_linear_id();
  const size_t       lrange     = g.get_local_linear_range();
  const unsigned int activemask = rv_mask();

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  size_t next_id = lid - 1;
  if (g.leader())
    next_id = 0;

  auto return_value = detail::shuffle_impl(local_x, next_id);

  if (g.leader())
    return init;

  return binary_op(return_value, init);
#else
  return binary_op(x, init);
#endif
}

// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  if (lid + 1 < 1024)
    scratch[lid + 1] = x;
  group_barrier(g);

  if (g.leader()) {
    scratch[0] = init;
    for (int i = 1; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1]);
  }

  group_barrier(g);
  T tmp = scratch[lid];
  group_barrier(g);

  return tmp;
}

// inclusive_scan
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  T *          scratch = static_cast<T *>(g.get_local_memory_ptr());
  const size_t lid     = g.get_local_linear_id();

  scratch[lid] = x;
  group_barrier(g);

  if (g.leader()) {
    for (int i = 1; i < g.get_local_range().size(); ++i)
      scratch[i] = binary_op(scratch[i], scratch[i - 1]);
  }

  group_barrier(g);
  T tmp = scratch[lid];
  group_barrier(g);

  return tmp;
}

template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(sub_group g, T x, BinaryOperation binary_op) {
#ifdef HIPSYCL_HAS_RV
  const size_t       lid        = g.get_local_linear_id();
  const size_t       lrange     = g.get_local_linear_range();
  const unsigned int activemask = rv_mask();

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::shuffle_impl(local_x, next_id);
    if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
#else
  return x;
#endif
}

// shift_left
template<typename Group, typename T>
T shift_group_left(Group g, T x, typename Group::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename Group::linear_id_type lid        = g.get_local_linear_id();
  typename Group::linear_id_type target_lid = lid + delta;

  scratch[lid] = x;
  group_barrier(g);

  if (target_lid > g.get_local_range().size())
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
T shift_group_left(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
#ifdef HIPSYCL_HAS_RV
  return detail::shuffle_down_impl(x, delta);
#else
  return x;
#endif
}

// shift_right
template<typename Group, typename T>
T shift_group_right(Group g, T x, typename Group::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename Group::linear_id_type lid        = g.get_local_linear_id();
  typename Group::linear_id_type target_lid = lid - delta;

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'Group::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
T shift_group_right(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
#ifdef HIPSYCL_HAS_RV
  return detail::shuffle_up_impl(x, delta);
#else
  return x;
#endif
}

// permute_group_by_xor
template<typename Group, typename T>
T permute_group_by_xor(Group g, T x, typename Group::linear_id_type mask) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename Group::linear_id_type lid        = g.get_local_linear_id();
  typename Group::linear_id_type target_lid = lid ^ mask;

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'Group::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

// permute_group_by_xor
template<typename T>
T permute_group_by_xor(sub_group g, T x, typename sub_group::linear_id_type mask) {
#ifdef HIPSYCL_HAS_RV
  return detail::shuffle_xor_impl(x, mask);
#else
  return x;
#endif
}

// select_from_group
template<typename Group, typename T>
T select_from_group(Group g, T x, typename Group::id_type remote_local_id) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename Group::linear_id_type lid = g.get_local_linear_id();
  typename Group::linear_id_type target_lid =
      detail::linear_id<g.dimensions>::get(remote_local_id, g.get_local_range());

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'Group::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
T select_from_group(sub_group g, T x, typename sub_group::id_type remote_local_id) {
#ifdef HIPSYCL_HAS_RV
  typename sub_group::linear_id_type target_lid = remote_local_id.get(0);
  return detail::shuffle_impl(x, target_lid);
#else
  return x;
#endif
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
