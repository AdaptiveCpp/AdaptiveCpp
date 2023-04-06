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
 *    this list of conditions and the following disclaimer.
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

#ifndef HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#include "../../backend.hpp"
#include "../../detail/data_layout.hpp"
#include "../../detail/thread_hierarchy.hpp"
#include "../../id.hpp"
#include "../../sub_group.hpp"
#include "../../vec.hpp"
#include "../../functional.hpp"
#include "warp_shuffle.hpp"
#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA || HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl::detail::hiplike_builtins {

namespace detail {

inline constexpr size_t max_group_size = 1024;

// reduce implementation
template<int Dim, typename T, typename BinaryOperation>
__device__
T __hipsycl_group_reduce(group<Dim> g, T x, BinaryOperation binary_op, T *scratch) {

  const auto lid = g.get_local_linear_id();
  const size_t lrange =
      (g.get_local_range().size() + __hipsycl_warp_size - 1) / __hipsycl_warp_size;
  sub_group sg{};

  x = detail::__hipsycl_reduce_over_sub_group(sg, x, binary_op);
  if (sg.leader())
    scratch[lid / __hipsycl_warp_size] = x;
  __hipsycl_group_barrier(g);

  if (lrange == 1)
    return scratch[0];

  if (g.get_local_range().size() / __hipsycl_warp_size != __hipsycl_warp_size) {
    size_t outputs = lrange;
    for (size_t i = (lrange + 1) / 2; i > 1; i = (i + 1) / 2) {
      if (lid < i && lid + i < outputs)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      outputs = outputs / 2 + outputs % 2;
      __hipsycl_group_barrier(g);
    }
    __hipsycl_group_barrier(g);
    return binary_op(scratch[0], scratch[1]);
  } else {
    if (lid < __hipsycl_warp_size)
      x = detail::__hipsycl_reduce_over_sub_group(sg, scratch[lid], binary_op);

    if (lid == 0)
      scratch[0] = x;

    __hipsycl_group_barrier(g);
  }
  return scratch[0];
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
sycl::group<Group::dimensions> create_nd_group_from_sp_group(Group g) {
#if !defined(HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO)
  return sycl::group<g.dimensions>{g.get_group_id(), g.get_local_range(), g.get_group_range()};
#else
  return sycl::group<g.dimensions>{};
#endif
}

} // namespace detail

// broadcast
// ND-range

template<typename Group, typename T, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id) {
  T result;

  // check if group is a group, if it is a subgruop use warp communicaton otherwise use shared memory
  if constexpr (std::is_same_v<sub_group, Group>) {
    result = detail::__hipsycl_shuffle_impl(x, local_linear_id);
  } else {
    __shared__ std::aligned_storage_t<sizeof(T), alignof(T)> scratch_storage;
    T *scratch = reinterpret_cast<T *>(&scratch_storage);
    const size_t lid = g.get_local_linear_id();

    if (lid == local_linear_id)
      scratch[0] = x;
    __hipsycl_group_barrier(g);

    result = scratch[0];
  }

  return result;
}

// any_of
// ND-range

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const size_t lrange = std::min(static_cast<size_t>(g.get_local_linear_range()),
      static_cast<size_t>(__hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z));
  const auto lid = g.get_local_linear_id();
  Ptr start_ptr = first + lid;

  bool local = false;

  for (Ptr p = start_ptr; p < last; p += lrange) {
    local |= pred(*p);
  }

  return any_of_group(g, local);
}

template<typename Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_any_of_group(Group g, bool pred) {
  bool result = pred;

  // check if Group is a group, since subgroups have special instruction
  if constexpr (std::is_same_v<sub_group, Group>) {
    result = detail::__hipsycl_any_of_sub_group(g, pred);
  } else {
    result = __syncthreads_or(pred);
  }

  return result;
}

// scoped V2
template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_any_of(Group g, Ptr first, Ptr last, Predicate pred) {
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_joint_any_of(nd_group, first, last, pred);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_joint_any_of(nd_subgroup, first, last, pred);
  }
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_any_of_group(Group g, const private_memory_access<bool, Group> &x) {
  bool tmp;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { tmp = x(idx); });
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_any_of_group(nd_group, tmp);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_any_of_group(nd_subgroup, tmp);
  }
}

// all_of
// ND-range

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  const size_t lrange = std::min(static_cast<size_t>(g.get_local_linear_range()),
      static_cast<size_t>(__hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z));
  const auto lid = g.get_local_linear_id();
  Ptr start_ptr = first + lid;

  bool local = true;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local &= pred(*p);

  return all_of_group(g, local);
}

template<typename Group, std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_all_of_group(Group g, bool pred) {
  bool result = pred;

  // check if Group is a group, since subgroups have special instruction
  if constexpr (std::is_same_v<sub_group, Group>) {
    result = detail::__hipsycl_all_of_sub_group(g, pred);
  } else {
    result = __syncthreads_and(pred);
  }

  return result;
}

// scoped V2

template<typename Group, typename Ptr, typename Predicate,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_joint_all_of(Group g, Ptr first, Ptr last, Predicate pred) {
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_joint_all_of(nd_group, first, last, pred);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_joint_all_of(nd_subgroup, first, last, pred);
  }
}

template<typename Group, std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
bool __hipsycl_all_of_group(Group g, const private_memory_access<bool, Group> &x) {
  bool tmp;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { tmp = x(idx); });
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_all_of_group(nd_group, tmp);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_all_of_group(nd_subgroup, tmp);
  }
}

// reduce
// ND-range

template<typename Group, typename Ptr, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type __hipsycl_joint_reduce(
    Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  using T = typename std::iterator_traits<Ptr>::value_type;

  const size_t lrange = std::min(static_cast<size_t>(g.get_local_linear_range()),
      static_cast<size_t>(__hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z));
  const size_t num_elements = last - first;
  const size_t lid = g.get_local_linear_id();

  Ptr start_ptr = first + lid;

  if (num_elements <= 0) {
    return known_identity_v<BinaryOperation, T>;
  }

  if (num_elements == 1)
    return *first;

  T local;

  // if we have more elements, than work items reduce
  if (num_elements >= lrange) {
    local = *start_ptr;

    for (Ptr p = start_ptr + lrange; p < last; p += lrange)
      local = binary_op(local, *p);
  }

  // set remaining elements in local to identity
  if (lid >= num_elements)
    local = known_identity_v<BinaryOperation, T>;

  return reduce_over_group(g, local, binary_op);
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_reduce_over_group(Group g, T x, BinaryOperation binary_op) {
  T result;

  // reduce normally if it is not a subgroup
  if constexpr (std::is_same_v<sub_group, Group>) {
    result = detail::__hipsycl_reduce_over_sub_group(g, x, binary_op);
  } else {
    __shared__
        std::aligned_storage_t<sizeof(T) * detail::max_group_size / __hipsycl_warp_size, alignof(T)>
            scratch_storage;
    T *scratch = reinterpret_cast<T *>(&scratch_storage);

    result = detail::__hipsycl_group_reduce(g, x, binary_op, scratch);
  }

  return result;
}

// scoped V2
template<typename Group, typename Ptr, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
typename std::iterator_traits<Ptr>::value_type __hipsycl_joint_reduce(
    Group g, Ptr first, Ptr last, BinaryOperation binary_op) {
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_joint_reduce(nd_group, first, last, binary_op);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_joint_reduce(nd_subgroup, first, last, binary_op);
  }
}

template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_reduce_over_group(
    Group g, const private_memory_access<T, Group> &x, BinaryOperation binary_op) {
  T tmp;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { tmp = x(idx); });
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_reduce_over_group(nd_group, tmp, binary_op);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_reduce_over_group(nd_subgroup, tmp, binary_op);
  }
}

// exclusive_scan
// ND-range

template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr __hipsycl_joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
  using V = typename std::iterator_traits<OutPtr>::value_type;

  const size_t lid = g.get_local_linear_id();
  const size_t lrange = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t num_iterations = (num_elements + lrange - 1) / lrange;

  // ensure correct return value
  if (num_elements <= 1)
    return first;

  result[0] = init;

  for (size_t i = 0; i < num_iterations; ++i) {
    // read value into tmp, we don't care about the value in case it's past the end
    V tmp;
    if (lid + i * lrange < num_elements)
      tmp = first[lid + i * lrange];

    tmp = inclusive_scan_over_group(g, tmp, binary_op, init);

    // write back correct values
    if (lid + i * lrange + 1 < num_elements)
      result[lid + i * lrange + 1] = tmp;

    // update init using the last partial result
    init = group_broadcast(g, tmp, lrange - 1);
  }

  return first + num_elements;
}

template<int Dim, typename T, typename V, typename BinaryOperation>
__device__
T __hipsycl_exclusive_scan_over_group(group<Dim> g, T x, V init, BinaryOperation binary_op) {
  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __hipsycl_warp_size, alignof(T)>
          scratch_storage;
  T *scratch = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid = g.get_local_linear_id();
  const size_t wid = lid / __hipsycl_warp_size;
  const size_t lrange = g.get_local_range().size();
  const size_t last_wid = (wid + 1) * __hipsycl_warp_size - 1;
  sub_group sg{};

  auto local_x = __hipsycl_inclusive_scan_over_group(sg, x, binary_op);

  // store carries
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  __hipsycl_group_barrier(g);

  // scan over carries
  if (lid < __hipsycl_warp_size) {
    const size_t scratch_index = (lid < detail::max_group_size / __hipsycl_warp_size) ? lid : 0;
    const T tmp = __hipsycl_inclusive_scan_over_group(sg, scratch[scratch_index], binary_op);

    if (lid < (lrange + __hipsycl_warp_size - 1) / __hipsycl_warp_size)
      scratch[lid] = tmp;
  }
  __hipsycl_group_barrier(g);

  // update result according to carry
  auto prefix = init;
  if (wid != 0)
    prefix = binary_op(init, scratch[wid - 1]);
  local_x = binary_op(prefix, local_x);

  local_x = detail::__hipsycl_shuffle_up_impl(local_x, 1);
  if (lid % __hipsycl_warp_size == 0)
    return prefix;
  return local_x;
}

// scoped V2

template<typename Group, typename InPtr, typename OutPtr, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
OutPtr __hipsycl_joint_exclusive_scan(
    Group g, InPtr first, InPtr last, OutPtr result, T init, BinaryOperation binary_op) {
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    return __hipsycl_joint_exclusive_scan(nd_group, first, last, result, init, binary_op);
  } else {
    sycl::sub_group nd_subgroup{};
    return __hipsycl_joint_exclusive_scan(nd_subgroup, first, last, result, init, binary_op);
  }
}

template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
void __hipsycl_exclusive_scan_over_group(Group g, const private_memory_access<V, Group> &x, T init,
    BinaryOperation binary_op, private_memory_access<T, Group> &result) {
  T tmp;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { tmp = x(idx); });
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    tmp = __hipsycl_exclusive_scan_over_group(nd_group, tmp, init, binary_op);
  } else {
    sycl::sub_group nd_subgroup{};
    tmp = __hipsycl_exclusive_scan_over_group(nd_subgroup, tmp, init, binary_op);
  }
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { result(idx) = tmp; });
}

// inclusive_scan
// ND-range
template<typename Group, typename T, typename BinaryOperation,
    std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
T __hipsycl_inclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __hipsycl_warp_size, alignof(T)>
          scratch_storage;
  T *scratch = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid = g.get_local_linear_id();
  size_t wid = lid / __hipsycl_warp_size;
  size_t lrange = g.get_local_range().size();
  size_t last_wid = (wid + 1) * __hipsycl_warp_size - 1;

  sub_group sg{};

  auto local_x = __hipsycl_inclusive_scan_over_group(sg, x, binary_op);
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  __hipsycl_group_barrier(g);

  if (lid < __hipsycl_warp_size) {
    size_t scratch_index =
        (lid < (lrange + __hipsycl_warp_size - 1) / __hipsycl_warp_size) ? lid : 0;
    const T tmp = __hipsycl_inclusive_scan_over_group(sg, scratch[scratch_index], binary_op);

    if (lid < detail::max_group_size / __hipsycl_warp_size) {
      scratch[lid] = tmp;
    }
  }
  __hipsycl_group_barrier(g);

  if (wid == 0)
    return local_x;
  return binary_op(scratch[wid - 1], local_x);
}

// scopedv2
template<typename Group, typename V, typename T, typename BinaryOperation,
    std::enable_if_t<is_sp_group_v<std::decay_t<Group>>, bool> = true>
HIPSYCL_KERNEL_TARGET
void __hipsycl_inclusive_scan_over_group(Group g, const private_memory_access<V, Group> &x,
    BinaryOperation binary_op, private_memory_access<T, Group> &result) {
  T tmp;
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { tmp = x(idx); });
  if constexpr (Group::fence_scope != memory_scope::sub_group) {
    sycl::group<g.dimensions> nd_group = detail::create_nd_group_from_sp_group(g);
    tmp = __hipsycl_inclusive_scan_over_group(nd_group, tmp, binary_op);
  } else {
    sycl::sub_group nd_subgroup{};
    tmp = __hipsycl_inclusive_scan_over_group(nd_subgroup, tmp, binary_op);
  }
  sycl::distribute_items(g, [&](sycl::s_item<1> idx) { result(idx) = tmp; });
}

// shift_left
template<typename T>
__device__
T __hipsycl_shift_group_left(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return detail::__hipsycl_shuffle_down_impl(x, delta);
}

// shift_right
template<typename T>
__device__
T __hipsycl_shift_group_right(sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return detail::__hipsycl_shuffle_up_impl(x, delta);
}

// permute_group_by_xor
template<typename T>
__device__
T __hipsycl_permute_group_by_xor(sub_group g, T x, typename sub_group::linear_id_type mask) {
  return detail::__hipsycl_shuffle_xor_impl(x, mask);
}

// select_from_group
template<typename T>
__device__
T __hipsycl_select_from_group(sub_group g, T x, typename sub_group::id_type remote_local_id) {
  typename sub_group::linear_id_type target_lid = remote_local_id.get(0);
  return detail::__hipsycl_shuffle_impl(x, target_lid);
}

} // namespace sycl::detail::hiplike_builtins
} // namespace hipsycl

#endif
#endif // HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
