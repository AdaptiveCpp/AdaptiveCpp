/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP

#include "backend.hpp"
#include "group_traits.hpp"
#include "group.hpp"
#include "sub_group.hpp"
#include "sp_group.hpp"
#include "vec.hpp"
#include "detail/builtin_dispatch.hpp"

#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
#include "cuda/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "hip/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA || HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "generic/hiplike/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
#include "spirv/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/group_functions.hpp"
#endif

#define HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(name, ...) \
  using namespace detail;                                  \
  HIPSYCL_RETURN_DISPATCH_BUILTIN(name, __VA_ARGS__);
#define HIPSYCL_DISPATCH_GROUP_ALGORITHM(name, ...) \
  using namespace detail;                           \
  HIPSYCL_DISPATCH_BUILTIN(name, __VA_ARGS__);

namespace hipsycl {
namespace sycl {


// broadcast
template<class Group, typename T, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_broadcast, g, x, local_linear_id);
}

// barrier
template<class Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void group_barrier(Group g) {
  HIPSYCL_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_barrier, g);
}

template<class Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void group_barrier(Group g, memory_scope fence_scope) {
  HIPSYCL_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_barrier, g, fence_scope);
}

// any_of
template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
bool joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_any_of, g, first, last, pred);
}

template<class Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, bool pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_any_of_group, g, pred);
}

template<typename Group, std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, const detail::private_memory_access<bool, Group> &pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_any_of_group, g, pred);
}


// all_of

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
bool joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_all_of, g, first, last, pred);
}

template<class Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, bool pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_all_of_group, g, pred);
}

template<typename Group, std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, const detail::private_memory_access<bool, Group> &pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_all_of_group, g, pred);
}


// reduce

template<typename Group, typename Ptr, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
typename std::iterator_traits<Ptr>::value_type joint_reduce(
    Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_reduce, g, first, last, binary_op);
}

template<class Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_reduce_over_group, g, x, binary_op);
}

template<class Group, typename T, typename BinaryOperation,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T reduce_over_group(
    Group g, const detail::private_memory_access<T, Group> &x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_reduce_over_group, g, x, binary_op);
}

// exclusive_scan
template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
OutPtr joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(
      __hipsycl_joint_exclusive_scan, g, first, last, result, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(
      __hipsycl_exclusive_scan_over_group, g, x, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void exclusive_scan_over_group(Group g, const detail::private_memory_access<V, Group> &x, T init,
    BinaryOperation binary_op, detail::private_memory_access<T, Group> &result) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(
      __hipsycl_exclusive_scan_over_group, g, x, init, binary_op, result);
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void exclusive_scan_over_group(Group g, const detail::private_memory_access<T, Group> &x,
    BinaryOperation binary_op, detail::private_memory_access<T, Group> &result) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(
      __hipsycl_exclusive_scan_over_group, g, x, binary_op, result);
}

// inclusive scan
template<class Group, class T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_inclusive_scan_over_group, g, x, binary_op);
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void inclusive_scan_over_group(Group g, const detail::private_memory_access<T, Group> &x,
    BinaryOperation binary_op, detail::private_memory_access<T, Group> &result) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(
      __hipsycl_inclusive_scan_over_group, g, x, binary_op, result);
}

// shift_left
template<class Group, typename T,
    std::enable_if_t<std::is_same_v<sub_group, std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T shift_group_left(Group g, T x, typename Group::linear_id_type delta = 1) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_shift_group_left, g, x, delta);
}

// shift_right
template<class Group, typename T,
    std::enable_if_t<std::is_same_v<sub_group, std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T shift_group_right(Group g, T x, typename Group::linear_id_type delta = 1) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_shift_group_right, g, x, delta);
}

// permute_group_by_xor
template<class Group, typename T,
    std::enable_if_t<std::is_same_v<sub_group, std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T permute_group_by_xor(Group g, T x, typename Group::linear_id_type mask) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_permute_group_by_xor, g, x, mask);
}


// select_from_group
template<class Group, typename T,
    std::enable_if_t<std::is_same_v<sub_group, std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T select_from_group(Group g, T x, typename Group::id_type remote_local_id) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_select_from_group, g, x, remote_local_id);
}

// ************* backend-independent overloads *********************

// barrier
template<class Group, typename T, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  typename Group::linear_id_type target_lid =
      hipsycl::sycl::detail::linear_id<g.dimensions>::get(local_id, g.get_local_range());

  return group_broadcast(g, x, target_lid);
}

// any_of
template<typename Group, typename T, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, T x, Predicate pred) {
  return any_of_group(g, pred(x));
}

template<typename Group, typename T, typename Predicate,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, const sycl::detail::private_memory_access<T, Group> &x, Predicate pred) {
  bool result;
  sycl::memory_environment(g, sycl::require_private_mem<bool>(), [&](auto &private_mem) {
    sycl::distribute_items(g, [&](sycl::s_item<1> idx) { private_mem(idx) = pred(x(idx)); });
    result = any_of_group(g, pred(x));
  });
  return result;
}

// all_of
template<typename Group, typename T, typename Predicate,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

template<typename Group, typename T, typename Predicate,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, const sycl::detail::private_memory_access<T, Group> &x, Predicate pred) {
  bool result;
  sycl::memory_environment(g, sycl::require_private_mem<bool>(), [&](auto &private_mem) {
    sycl::distribute_items(g, [&](sycl::s_item<1> idx) { private_mem(idx) = pred(x(idx)); });
    result = all_of_group(g, pred(x));
  });
  return result;
}

// none_of
template<typename Group, typename T, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool none_of_group(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

template<typename Group, typename T,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
bool none_of_group(Group g, T pred) {
  return !any_of_group(g, pred);
}

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
bool joint_none_of(Group g, Ptr first, Ptr last, Predicate pred) {
  return !joint_any_of(g, first, last, pred);
}

template<typename Group, typename T, typename Predicate,
    std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool none_of_group(
    Group g, const sycl::detail::private_memory_access<T, Group> &x, Predicate pred) {
  bool result;
  sycl::memory_environment(g, sycl::require_private_mem<bool>(), [&](auto &private_mem) {
    sycl::distribute_items(g, [&](sycl::s_item<1> idx) { private_mem(idx) = pred(x(idx)); });
    result = none_of_group(g, pred(x));
  });
  return result;
}

// reduce
template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_BUILTIN
T reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  T reduction = reduce_over_group(g, T{x}, binary_op);
  return binary_op(reduction, init);
}

template<typename Group, typename Ptr, typename T, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
T joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  T reduction = joint_reduce(g, first, last, binary_op);
  return binary_op(reduction, init);
}

// exclusive_scan
template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, known_identity_v<BinaryOperation, T>, binary_op);
}

template<typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op) {
  return joint_exclusive_scan(g, first, last, result,
      known_identity_v<BinaryOperation, typename std::iterator_traits<OutPtr>::value_type>,
      binary_op);
}

// inclusive scan
template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T inclusive_scan_over_group(Group g, V x, BinaryOperation binary_op, T init) {
  T y = x;
  if (g.leader()) {
    y = binary_op(x, init);
  }

  return inclusive_scan_over_group(g, y, binary_op);
}

template<typename Group, typename InPtr, typename OutPtr, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_inclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op) {
  // use exclusive scan with first element as init
  if (first >= last)
    return result;

  // last element is not read so last+1 is safe
  return joint_exclusive_scan(g, first + 1, last + 1, result, *first, binary_op);
}

template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<
        (is_group_v<std::decay_t<Group>> || detail::is_sp_group_v<std::decay_t<Group>>), bool> =
        true>
HIPSYCL_KERNEL_TARGET
OutPtr joint_inclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, BinaryOperation binary_op, T init) {
  if (first >= last)
    return result;

  // use exclusive scan with first element + init as init
  // last element is not read so last+1 is safe
  return joint_exclusive_scan(g, first + 1, last + 1, result, binary_op(init, *first), binary_op);
}


} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP
