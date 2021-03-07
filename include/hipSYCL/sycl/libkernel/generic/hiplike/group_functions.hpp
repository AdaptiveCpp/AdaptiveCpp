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

#ifdef SYCL_DEVICE_ONLY

#ifndef HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#include "../../backend.hpp"
#include "../../detail/data_layout.hpp"
#include "../../detail/thread_hierarchy.hpp"
#include "../../id.hpp"
#include "../../sub_group.hpp"
#include "../../vec.hpp"
#include "warp_shuffle.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

// reduce implementation
template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op, T *scratch) {

  const auto   lid    = g.get_local_linear_id();
  const size_t lrange = (g.get_local_range().size() + warpSize - 1) / warpSize;
  sub_group    sg{};

  x = group_reduce(sg, x, binary_op);
  if (sg.leader())
    scratch[lid / warpSize] = x;
  group_barrier(g);

  if (lrange == 1)
    return scratch[0];

  if (g.get_local_range().size() / warpSize != warpSize) {
    size_t outputs = lrange;
    for (size_t i = (lrange + 1) / 2; i > 1; i = (i + 1) / 2) {
      if (lid < i && lid + i < outputs)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      outputs = outputs / 2 + outputs % 2;
      group_barrier(g);
    }
    group_barrier(g);
    return binary_op(scratch[0], scratch[1]);
  } else {
    if (lid < warpSize)
      x = group_reduce(sg, scratch[lid], binary_op);

    if (lid == 0)
      scratch[0] = x;

    group_barrier(g);
  }
  return scratch[0];
}

// any_of
template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, T *first, T *last) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = false;

  for (T *p = start_ptr; p < last; p += lrange)
    local |= *p;

  return group_any_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool any_of(Group g, T *first, T *last, Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = false;

  for (T *p = start_ptr; p < last; p += lrange)
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
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = true;

  for (T *p = start_ptr; p < last; p += lrange)
    local &= *p;

  return group_all_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool all_of(Group g, T *first, T *last, Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = true;

  for (T *p = start_ptr; p < last; p += lrange)
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
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = false;

  for (T *p = start_ptr; p < last; p += lrange)
    local |= *p;

  return group_none_of(g, local);
}

template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool none_of(Group g, T *first, T *last, Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  T *        start_ptr = first + lid;

  bool local = false;

  for (T *p = start_ptr; p < last; p += lrange)
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
  __shared__ std::aligned_storage_t<sizeof(T) * 1024 / warpSize, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  const size_t lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t lid          = g.get_local_linear_id();

  T *start_ptr = first + lid;

  if (num_elements <= 0)
    return T{};

  if (num_elements == 1)
    return *first;

  if (num_elements >= lrange) {
    auto local = *start_ptr;

    for (T *p = start_ptr + lrange; p < last; p += lrange)
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
    group_barrier(g);

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
T leader_reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  return reduce(g, first, last, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
  auto         lid          = g.get_local_linear_id();
  auto         wid          = lid / warpSize;
  size_t       lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t iterations   = (num_elements + lrange - 1) / lrange;

  const size_t warp_size =
      (wid == lrange / warpSize && lrange % warpSize != 0) ? lrange % warpSize : warpSize;

  size_t offset     = lid;
  T      carry_over = init;
  T      local_x;

  for (int i = 0; i < iterations; ++i) {
    const size_t offset = lid + i * lrange;
    local_x             = (offset < num_elements) ? first[offset] : V{};
    local_x             = group_inclusive_scan(g, local_x, carry_over, binary_op);

    if (offset < num_elements)
      result[offset] = local_x;

    carry_over = group_broadcast(g, local_x, lrange - 1);
  }

  return result + num_elements;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *inclusive_scan(Group g, V *first, V *last, T *result, BinaryOperation binary_op) {
  auto         lid          = g.get_local_linear_id();
  auto         wid          = lid / warpSize;
  size_t       lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t iterations   = (num_elements + lrange - 1) / lrange;

  T carry_over;
  T local_x;

  for (int i = 0; i < iterations; ++i) {
    const size_t offset = lid + i * lrange;
    local_x             = (offset < num_elements) ? first[offset] : V{};
    if (i > 0) {
      local_x = group_inclusive_scan(g, local_x, carry_over, binary_op);
    } else {
      local_x = group_inclusive_scan(g, local_x, binary_op);
    }

    if (offset < num_elements)
      result[offset] = local_x;

    carry_over = group_broadcast(g, local_x, lrange - 1);
  }

  return result + num_elements;
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return inclusive_scan(g, first, last, result, binary_op);
}

// exclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *exclusive_scan(Group g, V *first, V *last, T *result, T init,
                  BinaryOperation binary_op) {
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
T *leader_exclusive_scan(Group g, V *first, V *last, T *result, T init,
                         BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, init, binary_op);
}

template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T *eleader_xclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return exclusive_scan(g, first, last, result, binary_op);
}

} // namespace detail

// broadcast
template<typename Group, typename T,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::linear_id_type local_linear_id = 0) {
  __shared__ std::aligned_storage_t<sizeof(T), alignof(T)> scratch_storage;
  T *          scratch = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid     = g.get_local_linear_id();

  if (lid == local_linear_id)
    scratch[0] = x;
  group_barrier(g);

  return scratch[0];
}

template<typename Group, typename T>
HIPSYCL_KERNEL_TARGET
T group_broadcast(Group g, T x, typename Group::id_type local_id) {
  const auto target_lid = detail::linear_id<g.dimensions>::get(
      local_id, detail::get_local_size<g.dimensions>());
  return group_broadcast(g, x, target_lid);
}

// any_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_any_of(Group g, bool pred) {
  return __syncthreads_or(pred);
}

// all_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_all_of(Group g, bool pred) {
  return __syncthreads_and(pred);
}

// none_of
template<typename Group>
HIPSYCL_KERNEL_TARGET
inline bool group_none_of(Group g, bool pred) {
  return !__syncthreads_or(pred);
}

// reduce
template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  __shared__ std::aligned_storage_t<sizeof(T) * 1024 / warpSize, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  return detail::group_reduce(g, x, binary_op, scratch);
}

// exclusive_scan
template<typename Group, typename T, typename V, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, T x, V init, BinaryOperation binary_op) {
  __shared__   std::aligned_storage_t<sizeof(T) * 1024 / warpSize, alignof(T)>
               scratch_storage;
  T *          scratch  = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid      = g.get_local_linear_id();
  const size_t wid      = lid / warpSize;
  const size_t lrange   = g.get_local_range().size();
  const size_t last_wid = (wid + 1) * warpSize - 1;
  sub_group    sg{};

  auto local_x = group_inclusive_scan(sg, x, binary_op);
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  group_barrier(g);

  if (lid < warpSize) {
    const size_t scratch_index = (lid < 1024 / warpSize) ? lid : 0;
    const T      tmp = group_inclusive_scan(sg, scratch[scratch_index], binary_op);

    if (lid < (lrange + warpSize - 1) / warpSize)
      scratch[lid] = tmp;
  }
  group_barrier(g);

  auto prefix = init;
  if (wid != 0)
    prefix = binary_op(init, scratch[wid - 1]);
  local_x = binary_op(prefix, local_x);

  local_x = detail::shuffle_up_impl(local_x, 1);
  if (lid % warpSize == 0)
    return prefix;
  return local_x;
}

// inclusive_scan
template<typename Group, typename T, typename BinaryOperation,
         typename std::enable_if_t<!std::is_same_v<Group, sub_group>, int> = 0>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  __shared__   std::aligned_storage_t<sizeof(T) * 1024 / warpSize, alignof(T)>
               scratch_storage;
  T *          scratch  = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid      = g.get_local_linear_id();
  size_t       wid      = lid / warpSize;
  size_t       lrange   = g.get_local_range().size();
  size_t       last_wid = (wid + 1) * warpSize - 1;

  sub_group sg{};

  auto local_x = group_inclusive_scan(sg, x, binary_op);
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  group_barrier(g);

  if (lid < warpSize) {
    size_t  scratch_index = (lid < (lrange + warpSize - 1) / warpSize) ? lid : 0;
    const T tmp           = group_inclusive_scan(sg, scratch[scratch_index], binary_op);

    if (lid < 1024 / warpSize) {
      scratch[lid] = tmp;
    }
  }
  group_barrier(g);

  if (wid == 0)
    return local_x;
  return binary_op(scratch[wid - 1], local_x);
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#endif // SYCL_DEVICE_ONLY
