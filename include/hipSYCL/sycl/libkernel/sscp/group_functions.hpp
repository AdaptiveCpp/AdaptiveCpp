/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021-2022 Aksel Alpay
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


#ifndef HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../detail/data_layout.hpp"
#include "../detail/mem_fence.hpp"
#include "../functional.hpp"
#include "../group.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

/// TODO: This file is a placeholder, most group algorithms are unimplemented!

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP

#include "builtins/barrier.hpp"

namespace hipsycl {
namespace sycl::detail::sscp_builtins {

// barrier
template <int Dim>
HIPSYCL_BUILTIN void
__hipsycl_group_barrier(group<Dim> g,
                        memory_scope fence_scope = group<Dim>::fence_scope) {
  __hipsycl_sscp_work_group_barrier(fence_scope, memory_order::seq_cst);
}

HIPSYCL_BUILTIN
 void
__hipsycl_group_barrier(sub_group g,
                        memory_scope fence_scope = sub_group::fence_scope) {
  __hipsycl_sscp_sub_group_barrier(fence_scope, memory_order::seq_cst);
}

// broadcast
template <int Dim, typename T>
HIPSYCL_BUILTIN 
T __hipsycl_group_broadcast(
    group<Dim> g, T x,
    typename group<Dim>::linear_id_type local_linear_id = 0);

template <int Dim, typename T>
HIPSYCL_BUILTIN T __hipsycl_group_broadcast(
    group<Dim> g, T x, typename group<Dim>::id_type local_id);

template<typename T>
HIPSYCL_BUILTIN
T __hipsycl_group_broadcast(sub_group g, T x,
                  typename sub_group::linear_id_type local_linear_id = 0);

template<typename T>
HIPSYCL_BUILTIN
T __hipsycl_group_broadcast(sub_group g, T x,
                  typename sub_group::id_type local_id);

// any_of

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN 
bool __hipsycl_joint_any_of(Group g, Ptr first, Ptr last,
                            Predicate pred);

template<int Dim>
HIPSYCL_BUILTIN
bool __hipsycl_any_of_group(group<Dim> g, bool pred);

HIPSYCL_BUILTIN
bool __hipsycl_any_of_group(sub_group g, bool pred);

// all_of

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN bool __hipsycl_joint_all_of(Group g, Ptr first, Ptr last,
                                                  Predicate pred);

template<int Dim>
HIPSYCL_BUILTIN
bool __hipsycl_all_of_group(group<Dim> g, bool pred);

HIPSYCL_BUILTIN
bool __hipsycl_all_of_group(sub_group g, bool pred);

// none_of

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN 
bool __hipsycl_joint_none_of(Group g, Ptr first, Ptr last,
                             Predicate pred);

template<int Dim>
HIPSYCL_BUILTIN
bool __hipsycl_none_of_group(group<Dim> g, bool pred);

HIPSYCL_BUILTIN
bool __hipsycl_none_of_group(sub_group g, bool pred);

// reduce

template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
typename std::iterator_traits<Ptr>::value_type
__hipsycl_joint_reduce(Group g, Ptr first, Ptr last, BinaryOperation binary_op);

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN
T __hipsycl_joint_reduce(Group g, Ptr first, Ptr last, T init,
                         BinaryOperation binary_op);

template<int Dim, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN
T __hipsycl_reduce_over_group(group<Dim> g, T x, BinaryOperation binary_op);

template<typename T, typename BinaryOperation>
HIPSYCL_BUILTIN
T __hipsycl_reduce_over_group(sub_group g, T x, BinaryOperation binary_op);

// exclusive_scan

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__hipsycl_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               T init, BinaryOperation binary_op);

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__hipsycl_joint_exclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op);

template <int Dim, typename V, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __hipsycl_exclusive_scan_over_group(
    group<Dim> g, V x, T init, BinaryOperation binary_op);


template <typename V, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __hipsycl_exclusive_scan_over_group(
    sub_group g, V x, T init, BinaryOperation binary_op);

template <typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN T
__hipsycl_exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op);

// inclusive_scan

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__hipsycl_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op, T init);

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN OutPtr
__hipsycl_joint_inclusive_scan(Group g, InPtr first, InPtr last, OutPtr result,
                               BinaryOperation binary_op);

template <int Dim, typename T, typename BinaryOperation>
HIPSYCL_BUILTIN
T __hipsycl_inclusive_scan_over_group(
    group<Dim> g, T x, BinaryOperation binary_op);

template <typename T, typename BinaryOperation>
HIPSYCL_BUILTIN T __hipsycl_inclusive_scan_over_group(
    sub_group g, T x, BinaryOperation binary_op);

template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_BUILTIN T __hipsycl_inclusive_scan_over_group(
    Group g, V x, T init, BinaryOperation binary_op);

// shift_left
template <int Dim, typename T>
HIPSYCL_BUILTIN
T __hipsycl_shift_group_left(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1);

template <typename T>
HIPSYCL_BUILTIN T __hipsycl_shift_group_left(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1);

// shift_right
template <int Dim, typename T>
HIPSYCL_BUILTIN T __hipsycl_shift_group_right(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1);

template <typename T>
HIPSYCL_BUILTIN T __hipsycl_shift_group_right(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1);

// permute_group_by_xor
template <int Dim, typename T>
HIPSYCL_BUILTIN T __hipsycl_permute_group_by_xor(
    group<Dim> g, T x, typename group<Dim>::linear_id_type mask);

// permute_group_by_xor
template <typename T>
HIPSYCL_BUILTIN T __hipsycl_permute_group_by_xor(
    sub_group g, T x, typename sub_group::linear_id_type mask);

// select_from_group
template <int Dim, typename T>
HIPSYCL_BUILTIN T __hipsycl_select_from_group(
    group<Dim> g, T x, typename group<Dim>::id_type remote_local_id);

template <typename T>
HIPSYCL_BUILTIN T __hipsycl_select_from_group(
    sub_group g, T x, typename sub_group::id_type remote_local_id);

} // namespace sycl::detail::sscp_builtins
} // namespace hipsycl

#endif

#endif // HIPSYCL_LIBKERNEL_SSCP_GROUP_FUNCTIONS_HPP

