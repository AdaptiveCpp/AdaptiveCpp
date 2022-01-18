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
#include "vec.hpp"
#include "detail/builtin_dispatch.hpp"

#include <type_traits>


#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
#include "cuda/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "hip/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "generic/hiplike/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
#include "spirv/group_functions.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/group_functions.hpp"
#endif

#define HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(name, ...)                     \
  using namespace detail;                                                      \
  HIPSYCL_RETURN_DISPATCH_BUILTIN(name, __VA_ARGS__);
#define HIPSYCL_DISPATCH_GROUP_ALGORITHM(name, ...)                            \
  using namespace detail;                                                      \
  HIPSYCL_DISPATCH_BUILTIN(name, __VA_ARGS__);

namespace hipsycl {
namespace sycl {


// broadcast
template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_broadcast, g, x,
                                          local_linear_id);
}

template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_broadcast, g, x,
                                          local_id);
}

// barrier
template<class Group,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void group_barrier(Group g) {
  HIPSYCL_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_barrier, g);
}

template<class Group,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
void group_barrier(Group g, memory_scope fence_scope) {
  HIPSYCL_DISPATCH_GROUP_ALGORITHM(__hipsycl_group_barrier, g, fence_scope);
}

// any_of
template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN bool joint_any_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_any_of, g, first,
                                          last, pred);
}

template<class Group,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, bool pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_any_of_group, g, pred);
}


// all_of

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN bool joint_all_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_all_of, g, first,
                                          last, pred);
}

template<class Group,
        std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, bool pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_all_of_group, g, pred);
}


// none_of

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN bool joint_none_of(Group g, Ptr first, Ptr last,
                                         Predicate pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_none_of, g, first,
                                          last, pred);
}

template<class Group,
         std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool none_of_group(Group g, bool pred) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_none_of_group, g, pred);
}


// reduce

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
typename std::iterator_traits<Ptr>::value_type
joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_reduce, g, first,
                                          last, binary_op);
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T joint_reduce(Group g, Ptr first, Ptr last, T init, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_reduce, g, first,
                                          last, init, binary_op);
}

template <class Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN T reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_reduce_over_group, g, x,
                                          binary_op);
}

// exclusive_scan
template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result, T init,
                       BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_exclusive_scan, g,
                                          first, last, result, init, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
OutPtr joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_exclusive_scan, g,
                                          first, last, result, binary_op);
}

template<class Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN
T exclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_exclusive_scan_over_group,
                                          g, x, init, binary_op);
}

template<typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_exclusive_scan_over_group,
                                          g, x, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                       BinaryOperation binary_op, T init) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_inclusive_scan, g,
                                          first, last, result, binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
OutPtr joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                            BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_joint_inclusive_scan, g,
                                          first, last, result, binary_op);
}

template<class Group, class T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_inclusive_scan_over_group,
                                          g, x, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T inclusive_scan_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_inclusive_scan_over_group,
                                          g, x, init, binary_op);
}

// shift_left
template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T shift_group_left(Group g, T x, typename Group::linear_id_type delta = 1) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_shift_group_left, g, x,
                                          delta);
}

// shift_right
template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T shift_group_right(Group g, T x, typename Group::linear_id_type delta = 1) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_shift_group_right, g, x,
                                          delta);
}

// permute_group_by_xor
template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T permute_group_by_xor(Group g, T x, typename Group::linear_id_type mask) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_permute_group_by_xor, g, x,
                                          mask);
}


// select_from_group
template<class Group, typename T,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T select_from_group(Group g, T x, typename Group::id_type remote_local_id) {
  HIPSYCL_RETURN_DISPATCH_GROUP_ALGORITHM(__hipsycl_select_from_group, g, x,
                                          remote_local_id);
}

// ************* backend-independent overloads *********************

// any_of
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool any_of_group(Group g, T x, Predicate pred) {
  return any_of_group(g, pred(x));
}

// all_of
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool all_of_group(Group g, T x, Predicate pred) {
  return all_of_group(g, pred(x));
}

// none_of
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
bool none_of_group(Group g, T x, Predicate pred) {
  return none_of_group(g, pred(x));
}

// reduce
template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T reduce_over_group(Group g, V x, T init, BinaryOperation binary_op) {
  T reduction = reduce_over_group(g, T{x}, binary_op);
  return binary_op(reduction, init);
}



} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP
