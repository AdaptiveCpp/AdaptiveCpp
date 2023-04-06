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


#ifndef HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../sp_group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include <type_traits>
#include <functional>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST

namespace hipsycl {
namespace sycl::detail::host_builtins {

// barrier
template<int Dim>
HIPSYCL_LOOP_SPLIT_BARRIER HIPSYCL_KERNEL_TARGET
inline void __hipsycl_group_barrier(
    group<Dim> g, memory_scope fence_scope = group<Dim>::fence_scope) {
  if (fence_scope == memory_scope::device) {
    mem_fence<>();
  }
  g.barrier();
}

HIPSYCL_KERNEL_TARGET
inline void __hipsycl_group_barrier(
    sub_group g, memory_scope fence_scope = sub_group::fence_scope) {
  // doesn't need sync
}

namespace detail {
// reduce implementation
template<int Dim, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T __hipsycl_group_reduce(group<Dim> g, T x, BinaryOperation binary_op, T *scratch) {
  const size_t lid = g.get_local_linear_id();

  scratch[lid] = x;
  __hipsycl_group_barrier(g);

  if (g.leader()) {
    T result = scratch[0];

    for (int i = 1; i < g.get_local_range().size(); ++i)
      result = binary_op(result, scratch[i]);

    scratch[0] = result;
  }

  __hipsycl_group_barrier(g);
  T tmp = scratch[0];
  __hipsycl_group_barrier(g);

  return tmp;
}

} // namespace detail

// broadcast
// ND-range

template<typename Group, typename T, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id) {
  T result = x;

  // check if Group is a group, since nothing needs to be done for subgroup
  if constexpr (!std::is_same_v<sub_group, Group>) {
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    if (g.get_local_linear_id() == local_linear_id)
      scratch[0] = x;

    __hipsycl_group_barrier(g);
    result = scratch[0];
    __hipsycl_group_barrier(g);
  }

  return result;
}

// any_of
// ND-range

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = false;

  if (g.leader()) {
    for (Ptr p = first; p < last; ++p) {
      result |= pred(*p);
    }
  }

  // broadcast result to all threads for non-sub_groups
  if constexpr (!std::is_same_v<sub_group, Group>) {
    result = group_broadcast(g, result);
  }

  return result;
}

template<typename Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_any_of_group(Group g, bool pred) {
  bool result = pred;

  // check if Group is a group, since nothing needs to be done for subgroup
  if constexpr (!std::is_same_v<sub_group, Group>) {
    bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

    scratch[0] = false;
    __hipsycl_group_barrier(g);

    // only write if a pred evaluates to true, because of the if-clause order doesn't matter
    if (result)
      scratch[0] = true;

    // read value from shared memory
    __hipsycl_group_barrier(g);
    result = scratch[0];
    __hipsycl_group_barrier(g);
  }

  return result;
}

// scoped V2
template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = false;

  single_item(g, [&]() {
    for (Ptr p = first; p < last; ++p) {
      result |= pred(*p);
    }
  });

  return result;
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_any_of_group(Group g, const private_memory_access<bool, Group> &x) {
  bool result = false;

  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { result |= x(idx); });

  return result;
}

// all_of
// ND-range

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  if (g.leader()) {
    for (Ptr p = first; p < last; ++p) {
      result &= pred(*p);
    }
  }

  // broadcast result to all threads for non-sub_groups
  if constexpr (!std::is_same_v<sub_group, Group>) {
    result = group_broadcast(g, result);
  }

  return result;
}

template<typename Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_all_of_group(Group g, bool pred) {
  bool result = pred;

  // check if Group is a group, since nothing needs to be done for subgroup
  if constexpr (!std::is_same_v<sub_group, Group>) {
    bool *scratch = static_cast<bool *>(g.get_local_memory_ptr());

    scratch[0] = true;
    __hipsycl_group_barrier(g);

    // only write if a pred evaluates to false, because of the if-clause order doesn't matter
    if (!result)
      scratch[0] = false;

    // read value from shared memory
    __hipsycl_group_barrier(g);
    result = scratch[0];
    __hipsycl_group_barrier(g);
  }

  return result;
}

// scoped V2

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  bool result = true;

  single_item(g, [&]() {
    for (Ptr p = first; p < last; ++p) {
      result &= pred(*p);
    }
  });

  return result;
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_all_of_group(Group g, const private_memory_access<bool, Group> &pred) {
  bool result = true;

  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { result &= pred(idx); });

  return result;
}

// reduce
// ND-range

template<typename Group, typename Ptr, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type __hipsycl_joint_reduce(
    Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  using T = typename std::iterator_traits<Ptr>::value_type;
  T result = known_identity_v<BinaryOperation, typename std::iterator_traits<Ptr>::value_type>;

  if (first >= last) {
    return result;
  }

  if (g.leader()) {
    for (T *i = first; i < last; ++i)
      result = binary_op(result, *i);
  }

  return group_broadcast(g, result);
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  T result = x;

  if constexpr (!std::is_same_v<sub_group, Group>) {
    const size_t lid = g.get_local_linear_id();
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    scratch[lid] = x;
    __hipsycl_group_barrier(g);

    if (g.leader()) {
      for (int i = 1; i < g.get_local_range().size(); ++i)
        result = binary_op(result, scratch[i]);

      scratch[0] = result;
    }

    __hipsycl_group_barrier(g);
    result = scratch[0];
    __hipsycl_group_barrier(g);
  }

  return result;
}

// scoped V2
template<typename Group, typename Ptr, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type __hipsycl_joint_reduce(
    Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  if (first >= last)
    return typename std::iterator_traits<Ptr>::value_type{};

  typename std::iterator_traits<Ptr>::value_type result = *first;

  single_item(g, [&]() {
    for (Ptr p = first + 1; p < last; ++p) {
      result = binary_op(result, *p);
    }
  });

  return result;
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_reduce_over_group(
    Group g, const private_memory_access<T, Group> &x, BinaryOperation binary_op) {
  T result = known_identity_v<BinaryOperation, T>;

  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { result = binary_op(result, x(idx)); });

  return result;
}

// exclusive_scan
// ND-range

template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr __hipsycl_joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
  // make sure the return value is correct
  if ((last - first) < 0) {
    return result;
  }

  if (g.leader()) {
    T tmp = init;
    *result = tmp;
    for (T *i = first; i < last - 1; ++i) {
      result++;
      tmp = binary_op(tmp, *i);
      *result = tmp;
    }
  }

  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  T result = init;

  if constexpr (!std::is_same_v<sub_group, Group>) {
    const size_t lid = g.get_local_linear_id();
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    scratch[lid] = x;
    __hipsycl_group_barrier(g);

    if (g.leader()) {
      for (int i = 0; i < g.get_local_range().size() - 1; ++i) {
        result = binary_op(result, scratch[i]);
        scratch[i] = result;
      }
    }

    __hipsycl_group_barrier(g);
    if (lid == 0) {
      result = init;
    } else {
      result = scratch[lid - 1];
    }
    __hipsycl_group_barrier(g);
  }

  return result;
}

// scoped V2

template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr __hipsycl_joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
  single_item(g, [&]() {
    *result = init;
    result++;
    for (InPtr p = first; p < last - 1; ++p) {
      *result = binary_op(*(result - 1), *p);
      result++;
    }
  });

  return result;
}

template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
void __hipsycl_exclusive_scan_over_group(Group g, const private_memory_access<V, Group> &x, T init,
    BinaryOperation binary_op, private_memory_access<T, Group> &result) {

  T partial = init;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
    T tmp = partial;
    partial = binary_op(partial, x(idx));
    result(idx) = tmp;
  });
}

// inclusive_scan
// ND-range
template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  T result = x;

  if constexpr (!std::is_same_v<sub_group, Group>) {
    const size_t lid = g.get_local_linear_id();
    T *scratch = static_cast<T *>(g.get_local_memory_ptr());

    scratch[lid] = x;
    __hipsycl_group_barrier(g);

    if (g.leader()) {
      for (int i = 1; i < g.get_local_range().size(); ++i) {
        result = binary_op(result, scratch[i]);
        scratch[i] = result;
      }
    }

    __hipsycl_group_barrier(g);
    result = scratch[lid];
    __hipsycl_group_barrier(g);
  }

  return result;
}

// scopedv2
template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
void __hipsycl_inclusive_scan_over_group(Group g, const private_memory_access<V, Group> &x,
    BinaryOperation binary_op, private_memory_access<T, Group> &result) {

  T partial = known_identity_v<BinaryOperation, T>;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
    partial = binary_op(partial, x(idx));
    result(idx) = partial;
  });
}

// shift_left
template<typename T>
HIPSYCL_KERNEL_TARGET
T __hipsycl_shift_group_left(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return x;
}

// shift_right
template<typename T>
HIPSYCL_KERNEL_TARGET
T __hipsycl_shift_group_right(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return x;
}

// permute_group_by_xor
template<typename T>
HIPSYCL_KERNEL_TARGET
T __hipsycl_permute_group_by_xor(sub_group g, T x, typename sub_group::linear_id_type mask) {
  return x;
}

// select_from_group
template<typename T>
HIPSYCL_KERNEL_TARGET
T __hipsycl_select_from_group(sub_group g, T x, typename sub_group::id_type remote_local_id) {
  return x;
}

} // namespace sycl::detail::host_builtins
} // namespace hipsycl

#endif

#endif // HIPSYCL_LIBKERNEL_HOST_GROUP_FUNCTIONS_HPP
