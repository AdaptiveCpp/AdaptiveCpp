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
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {
// reduce implementation
template<int Dim, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(group<Dim> g, T x, BinaryOperation binary_op, T *scratch) {
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
}

} // namespace detail

// broadcast
template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(group<Dim> g, T x, typename group<Dim>::linear_id_type local_linear_id = 0) {
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

template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(group<Dim> g, T x, typename group<Dim>::id_type local_id) {
  const size_t target_lid =
      detail::linear_id<g.dimensions>::get(local_id, g.get_local_range());
  return group_broadcast(g, x, target_lid);
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
  return x;
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(sub_group g, T x,
                  typename sub_group::id_type local_id) {
  return x;
}

// barrier
template<int Dim>
HIPSYCL_KERNEL_TARGET
inline void group_barrier(group<Dim> g, memory_scope fence_scope = group<Dim>::fence_scope) {
  if (fence_scope == memory_scope::device) {
    detail::mem_fence<>();
  }
  g.barrier();
}

HIPSYCL_KERNEL_TARGET
inline void group_barrier(sub_group g, memory_scope fence_scope) {
  // doesn't need sync
}

// any_of
namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
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
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET bool joint_any_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  const bool result = detail::leader_any_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template<int Dim>
HIPSYCL_KERNEL_TARGET
inline bool any_of_group(group<Dim> g, bool pred) {
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

HIPSYCL_KERNEL_TARGET
inline bool any_of_group(sub_group g, bool pred) {
  return pred;
}

// all_of
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
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
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET bool joint_all_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  const bool result = detail::leader_all_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template<int Dim>
HIPSYCL_KERNEL_TARGET
inline bool all_of_group(group<Dim> g, bool pred) {
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

HIPSYCL_KERNEL_TARGET
inline bool all_of_group(sub_group g, bool pred) {
  return pred;
}

// none_of
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename Ptr, typename Predicate,
        std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
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
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET bool joint_none_of(Group g, Ptr first, Ptr last,
                                         Predicate pred) {
  auto result = detail::leader_none_of(g, first, last, pred);
  return group_broadcast(g, result);
}

template<int Dim>
HIPSYCL_KERNEL_TARGET
inline bool none_of_group(group<Dim> g, bool pred) {
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

HIPSYCL_KERNEL_TARGET
inline bool none_of_group(sub_group g, bool pred) {
  return pred;
}

// reduce
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
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

template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T leader_reduce(Group g, T *first, T *last, V init, BinaryOperation binary_op) {
  auto result = leader_reduce(g, first, last, binary_op);

  if (g.leader()) {
    result = binary_op(result, init);
  }
  return result;
}
}

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  const auto result = detail::leader_reduce(g, first, last, binary_op);

  return group_broadcast(g, result);
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  const auto result = detail::leader_reduce(g, first, last, init, binary_op);

  return group_broadcast(g, result);
}

template<int Dim, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce_over_group(group<Dim> g, T x, BinaryOperation binary_op) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  T tmp = detail::group_reduce(g, x, binary_op, scratch);

  return tmp;
}

template<typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T reduce_over_group(sub_group g, T x, BinaryOperation binary_op) {
  return x;
}

// exclusive_scan
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
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

template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T *leader_exclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_exclusive_scan(g, first, last, result, T{}, binary_op);
}
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                       BinaryOperation binary_op) {
  const auto ret = detail::leader_exclusive_scan(g, first, last, result, init, binary_op);
  return group_broadcast(g, ret);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result, typename std::remove_pointer_t<InPtr>{}, binary_op);
}

template<int Dim, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T exclusive_scan_over_group(group<Dim> g, V x, T init, BinaryOperation binary_op) {
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

template<typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T exclusive_scan_over_group(sub_group g, V x, T init, BinaryOperation binary_op) {
  return binary_op(x, init);
}

// inclusive_scan
namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op, T init) {
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

template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return leader_inclusive_scan(g, first, last, result, binary_op, T{});
}
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                       BinaryOperation binary_op, T init) {
  auto ret = detail::leader_inclusive_scan(g, first, last, result, binary_op, init);
  return group_broadcast(g, ret);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  return joint_inclusive_scan(g, first, last, result, binary_op, typename std::remove_pointer_t<InPtr>{});
}
template<int Dim, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T inclusive_scan_over_group(group<Dim> g, T x, BinaryOperation binary_op) {
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
T inclusive_scan_over_group(sub_group g, T x, BinaryOperation binary_op) {
  return x;
}

// shift_left
template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T shift_group_left(group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid + delta;

  scratch[lid] = x;
  group_barrier(g);

  if (target_lid > g.get_local_range().size())
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T shift_group_left(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return x;
}

// shift_right
template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T shift_group_right(group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid - delta;

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T shift_group_right(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return x;
}

// permute_group_by_xor
template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T permute_group_by_xor(group<Dim> g, T x, typename group<Dim>::linear_id_type mask) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid ^ mask;

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

// permute_group_by_xor
template<typename T>
HIPSYCL_KERNEL_TARGET
T permute_group_by_xor(sub_group g, T x, typename sub_group::linear_id_type mask) {
  return x;
}

// select_from_group
template<int Dim, typename T>
HIPSYCL_KERNEL_TARGET
T select_from_group(group<Dim> g, T x, typename group<Dim>::id_type remote_local_id) {
  T *scratch = static_cast<T *>(g.get_local_memory_ptr());

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid =
      detail::linear_id<g.dimensions>::get(remote_local_id, g.get_local_range());

  scratch[lid] = x;
  group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  x = scratch[target_lid];
  group_barrier(g);

  return x;
}

template<typename T>
HIPSYCL_KERNEL_TARGET
T select_from_group(sub_group g, T x, typename sub_group::id_type remote_local_id) {
  return x;
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
