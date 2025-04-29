/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#ifndef ACPP_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP
#define ACPP_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

#include "../../backend.hpp"
#include "../../detail/data_layout.hpp"
#include "../../detail/thread_hierarchy.hpp"
#include "../../id.hpp"
#include "../../sub_group.hpp"
#include "../../vec.hpp"
#include "warp_shuffle.hpp"
#include <type_traits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl::detail::hiplike_builtins {

namespace detail {

inline constexpr size_t max_group_size = 1024;

// reduce implementation
template <int Dim, typename T, typename BinaryOperation>
__device__ T __acpp_group_reduce(group<Dim> g, T x,
                                    BinaryOperation binary_op, T *scratch) {

  const auto   lid    = g.get_local_linear_id();
  const size_t lrange = (g.get_local_range().size() + __acpp_warp_size - 1) / __acpp_warp_size;
  sub_group    sg{};

  x = __acpp_reduce_over_group(sg, x, binary_op);
  if (sg.leader())
    scratch[lid / __acpp_warp_size] = x;
  __acpp_group_barrier(g);

  if (lrange == 1)
    return scratch[0];

  if (g.get_local_range().size() / __acpp_warp_size != __acpp_warp_size) {
    size_t outputs = lrange;
    for (size_t i = (lrange + 1) / 2; i > 1; i = (i + 1) / 2) {
      if (lid < i && lid + i < outputs)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      outputs = outputs / 2 + outputs % 2;
      __acpp_group_barrier(g);
    }
    __acpp_group_barrier(g);
    return binary_op(scratch[0], scratch[1]);
  } else {
    if (lid < __acpp_warp_size)
      x = __acpp_reduce_over_group(sg, scratch[lid], binary_op);

    if (lid == 0)
      scratch[0] = x;

    __acpp_group_barrier(g);
  }
  return scratch[0];
}

} // namespace detail

// broadcast
template <int Dim, typename T>
__device__ T __acpp_group_broadcast(
    group<Dim> g, T x,
    typename group<Dim>::linear_id_type local_linear_id = 0) {
  __shared__ std::aligned_storage_t<sizeof(T), alignof(T)> scratch_storage;
  T *          scratch = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid     = g.get_local_linear_id();

  if (lid == local_linear_id)
    scratch[0] = x;
  __acpp_group_barrier(g);

  return scratch[0];
}

template <int Dim, typename T>
__device__ T __acpp_group_broadcast(group<Dim> g, T x,
                                       typename group<Dim>::id_type local_id) {
  const auto target_lid = linear_id<g.dimensions>::get(
      local_id, get_local_size<g.dimensions>());
  return __acpp_group_broadcast(g, x, target_lid);
}

template<typename T>
__device__
T __acpp_group_broadcast(sub_group sg, T x,
                  typename sub_group::linear_id_type local_linear_id = 0) {
  return detail::__acpp_shuffle_impl(x, static_cast<int>(local_linear_id));
}

template<typename T>
__device__
T __acpp_group_broadcast(sub_group sg, T x,
                            typename sub_group::id_type local_id) {
  const size_t target_lid =
      linear_id<1>::get(local_id, sg.get_local_range());
  return detail::__acpp_shuffle_impl(x, target_lid);
}

// any_of
template<int Dim>
__device__
inline bool __acpp_any_of_group(group<Dim> g, bool pred) {
  return __syncthreads_or(pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
bool __acpp_joint_any_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = false;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local |= pred(*p);

  return __acpp_any_of_group(g, local);
}

namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
bool __acpp_leader_any_of(Group g, T *first, T *last, Predicate pred) {
  return __acpp_any_of(g, first, last, pred);
}
}

// all_of
template<int Dim>
__device__
inline bool __acpp_all_of_group(group<Dim> g, bool pred) {
  return __syncthreads_and(pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ bool __acpp_joint_all_of(Group g, Ptr first, Ptr last,
                                      Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = true;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local &= pred(*p);

  return __acpp_all_of_group(g, local);
}

namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
bool __acpp_leader_all_of(Group g, T *first, T *last, Predicate pred) {
  return __acpp_all_of(g, first, last, pred);
}
}


// none_of
template<int Dim>
__device__
inline bool __acpp_none_of_group(group<Dim> g, bool pred) {
  return !__syncthreads_or(pred);
}

template <typename Group, typename Ptr, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ bool __acpp_joint_none_of(Group g, Ptr first, Ptr last,
                                        Predicate pred) {
  const auto lrange    = g.get_local_range().size();
  const auto lid       = g.get_local_linear_id();
  Ptr        start_ptr = first + lid;

  bool local = false;

  for (Ptr p = start_ptr; p < last; p += lrange)
    local |= pred(*p);

  return __acpp_none_of_group(g, local);
}

namespace detail { // until scoped-parallelism can be detected
template<typename Group, typename T, typename Predicate,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
bool __acpp_leader_none_of(Group g, T *first, T *last, Predicate pred) {
  return __acpp_none_of(g, first, last, pred);
}
}

// reduce
template <typename Group, typename Ptr, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ typename std::iterator_traits<Ptr>::value_type
__acpp_joint_reduce(Group g, Ptr first, Ptr last,
                       BinaryOperation binary_op) {
  using T = std::remove_pointer_t<Ptr>;

  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __acpp_warp_size, alignof(T)>
          scratch_storage;
  Ptr     scratch = reinterpret_cast<Ptr>(&scratch_storage);

  const size_t lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t lid          = g.get_local_linear_id();

  Ptr start_ptr = first + lid;

  if (num_elements <= 0)
    return T{};

  if (num_elements == 1)
    return *first;

  if (num_elements >= lrange) {
    auto local = *start_ptr;

    for (Ptr p = start_ptr + lrange; p < last; p += lrange)
      local = binary_op(local, *p);

    return detail::__acpp_group_reduce(g, local, binary_op, scratch);
  } else {
    const size_t warp_id           = lid / __acpp_warp_size;
    const size_t num_warps         = num_elements / __acpp_warp_size;
    const size_t elements_per_warp = (num_elements + num_warps - 1) / num_warps;
    size_t       outputs           = num_elements;

    if (num_warps == 0 && lid < num_elements) {
      scratch[lid] = first[lid];
    } else if (warp_id < num_warps) {
      const size_t active_threads = num_warps * __acpp_warp_size;

      auto local = *start_ptr;

      for (Ptr p = start_ptr + active_threads; p < last; p += active_threads)
        local = binary_op(local, *p);

      sub_group sg{};

      local = __acpp_reduce_over_group(sg, local, binary_op);
      if (sg.leader())
        scratch[warp_id] = local;
      outputs = num_warps;
    }
    __acpp_group_barrier(g);

    for (size_t i = (outputs + 1) / 2; i > 1; i = (i + 1) / 2) {
      if (lid < i && lid + i < outputs)
        scratch[lid] = binary_op(scratch[lid], scratch[lid + i]);
      outputs = outputs / 2 + outputs % 2;
      __acpp_group_barrier(g);
    }
    __acpp_group_barrier(g);

    return binary_op(scratch[0], scratch[1]);
  }
}

template <typename Group, typename Ptr, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T __acpp_joint_reduce(Group g, Ptr first, Ptr last, T init,
                                    BinaryOperation binary_op) {
  return binary_op(__acpp_joint_reduce(g, first, last, binary_op), init);
}

namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T __acpp_leader_reduce(Group g, V *first, V *last, T init,
                                     BinaryOperation binary_op) {
  return __acpp_joint_reduce(g, first, last, init, binary_op);
}

template<typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
T __acpp_leader_reduce(Group g, T *first, T *last, BinaryOperation binary_op) {
  return __acpp_joint_reduce(g, first, last, binary_op);
}
}

template<int Dim, typename T, typename BinaryOperation>
__device__
T __acpp_reduce_over_group(group<Dim> g, T x, BinaryOperation binary_op) {
  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __acpp_warp_size, alignof(T)>
          scratch_storage;
  T *     scratch = reinterpret_cast<T *>(&scratch_storage);

  return detail::__acpp_group_reduce(g, x, binary_op, scratch);
}

// inclusive_scan

template <int Dim, typename T, typename BinaryOperation>
__device__ T __acpp_inclusive_scan_over_group(group<Dim> g, T x,
                                                 BinaryOperation binary_op) {
  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __acpp_warp_size, alignof(T)>
               scratch_storage;
  T *          scratch  = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid      = g.get_local_linear_id();
  size_t       wid      = lid / __acpp_warp_size;
  size_t       lrange   = g.get_local_range().size();
  size_t       last_wid = (wid + 1) * __acpp_warp_size - 1;

  sub_group sg{};

  auto local_x = __acpp_inclusive_scan_over_group(sg, x, binary_op);
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  __acpp_group_barrier(g);

  if (lid < __acpp_warp_size) {
    size_t  scratch_index = (lid < (lrange + __acpp_warp_size - 1) / __acpp_warp_size) ? lid : 0;
    const T tmp =
        __acpp_inclusive_scan_over_group(sg, scratch[scratch_index], binary_op);

    if (lid < detail::max_group_size / __acpp_warp_size) {
      scratch[lid] = tmp;
    }
  }
  __acpp_group_barrier(g);

  if (wid == 0)
    return local_x;
  return binary_op(scratch[wid - 1], local_x);
}

template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T __acpp_inclusive_scan_over_group(Group g, V x, T init,
                                                 BinaryOperation binary_op) {
  T scan = __acpp_inclusive_scan_over_group(g, T{x}, binary_op);
  return binary_op(scan, init);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ OutPtr __acpp_joint_inclusive_scan(Group g, InPtr first,
                                                 InPtr last, OutPtr result,
                                                 BinaryOperation binary_op,
                                                 T init) {
  using OutT = std::remove_reference_t<decltype(*result)>;

  auto         lid          = g.get_local_linear_id();
  auto         wid          = lid / __acpp_warp_size;
  size_t       lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t iterations   = (num_elements + lrange - 1) / lrange;

  const size_t warp_size =
      (wid == lrange / __acpp_warp_size && lrange % __acpp_warp_size != 0) ? lrange % __acpp_warp_size : __acpp_warp_size;

  size_t offset     = lid;
  OutT      carry_over = init;
  OutT      local_x;

  for (int i = 0; i < iterations; ++i) {
    const size_t offset = lid + i * lrange;
    local_x = (offset < num_elements) ? first[offset] : OutT{};
    local_x =
        __acpp_inclusive_scan_over_group(g, local_x, carry_over, binary_op);

    if (offset < num_elements)
      result[offset] = local_x;

    carry_over = __acpp_group_broadcast(g, local_x, lrange - 1);
  }

  return result + num_elements;
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ OutPtr __acpp_joint_inclusive_scan(Group g, InPtr first,
                                                 InPtr last, OutPtr result,
                                                 BinaryOperation binary_op) {
  using OutT = std::remove_reference_t<decltype(*result)>;

  auto         lid          = g.get_local_linear_id();
  auto         wid          = lid / __acpp_warp_size;
  size_t       lrange       = g.get_local_range().size();
  const size_t num_elements = last - first;
  const size_t iterations   = (num_elements + lrange - 1) / lrange;

  OutT carry_over;
  OutT local_x;

  for (int i = 0; i < iterations; ++i) {
    const size_t offset = lid + i * lrange;
    local_x             = (offset < num_elements) ? first[offset] : OutT{};
    if (i > 0) {
      local_x =
          __acpp_inclusive_scan_over_group(g, local_x, carry_over, binary_op);
    } else {
      local_x = __acpp_inclusive_scan_over_group(g, local_x, binary_op);
    }

    if (offset < num_elements)
      result[offset] = local_x;

    carry_over = __acpp_group_broadcast(g, local_x, lrange - 1);
  }

  return result + num_elements;
}

namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T *
__acpp_leader_inclusive_scan(Group g, V *first, V *last, T *result,
                                BinaryOperation binary_op, T init) {
  return __acpp_joint_inclusive_scan(g, first, last, result, binary_op,
                                        init);
}

template<typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
T *__acpp_leader_inclusive_scan(Group g, V *first, V *last, T *result,
                         BinaryOperation binary_op) {
  return __acpp_joint_inclusive_scan(g, first, last, result, binary_op);
}
}

// exclusive_scan

template <int Dim, typename T, typename V, typename BinaryOperation>
__device__ T __acpp_exclusive_scan_over_group(group<Dim> g, T x, V init,
                                                 BinaryOperation binary_op) {
  __shared__
      std::aligned_storage_t<sizeof(T) * detail::max_group_size / __acpp_warp_size, alignof(T)>
               scratch_storage;
  T *          scratch  = reinterpret_cast<T *>(&scratch_storage);
  const size_t lid      = g.get_local_linear_id();
  const size_t wid      = lid / __acpp_warp_size;
  const size_t lrange   = g.get_local_range().size();
  const size_t last_wid = (wid + 1) * __acpp_warp_size - 1;
  sub_group    sg{};

  auto local_x = __acpp_inclusive_scan_over_group(sg, x, binary_op);
  if (lid == (lrange < last_wid ? lrange - 1 : last_wid))
    scratch[wid] = local_x;
  __acpp_group_barrier(g);

  if (lid < __acpp_warp_size) {
    const size_t scratch_index = (lid < detail::max_group_size / __acpp_warp_size) ? lid : 0;
    const T tmp =
        __acpp_inclusive_scan_over_group(sg, scratch[scratch_index], binary_op);

    if (lid < (lrange + __acpp_warp_size - 1) / __acpp_warp_size)
      scratch[lid] = tmp;
  }
  __acpp_group_barrier(g);

  auto prefix = init;
  if (wid != 0)
    prefix = binary_op(init, scratch[wid - 1]);
  local_x = binary_op(prefix, local_x);

  local_x = detail::__acpp_shuffle_up_impl(local_x, 1);
  if (lid % __acpp_warp_size == 0)
    return prefix;
  return local_x;
}

template<typename Group, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__
T __acpp_exclusive_scan_over_group(Group g, T x, BinaryOperation binary_op) {
  return __acpp_exclusive_scan_over_group(g, x, T{}, binary_op);
}

template <typename Group, typename InPtr, typename OutPtr, typename T,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ OutPtr __acpp_joint_exclusive_scan(Group g, InPtr first,
                                                 InPtr last, OutPtr result,
                                                 T init,
                                                 BinaryOperation binary_op) {
  auto lid = g.get_local_linear_id();
  if (lid == 0 && last - first > 0)
    result[0] = init;
  return __acpp_joint_inclusive_scan(g, first, last - 1, result + 1,
                                        binary_op, init);
}

template <typename Group, typename InPtr, typename OutPtr,
          typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ OutPtr __acpp_joint_exclusive_scan(Group g, InPtr first,
                                                 InPtr last, OutPtr result,
                                                 BinaryOperation binary_op) {
  using OutT = std::remove_reference_t<decltype(*result)>;
  return __acpp_joint_exclusive_scan(
      g, first, last, result, OutT{},
      binary_op);
}

namespace detail { // until scoped-parallelism can be detected
template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T *__acpp_leader_exclusive_scan(Group g, V *first, V *last,
                                              T *result, T init,
                                              BinaryOperation binary_op) {
  return __acpp_joint_exclusive_scan(g, first, last, result, init,
                                        binary_op);
}

template <typename Group, typename V, typename T, typename BinaryOperation,
          std::enable_if_t<is_group_v<std::decay_t<Group>>, bool> = true>
__device__ T *__acpp_leader_exclusive_scan(Group g, V *first, V *last,
                                              T *result,
                                              BinaryOperation binary_op) {
  return __acpp_joint_exclusive_scan(g, first, last, result, binary_op);
}
}

// shift_left
template <int Dim, typename T>
__device__ T __acpp_shift_group_left(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  __shared__ std::aligned_storage_t<sizeof(T) * detail::max_group_size, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid + delta;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  if (target_lid > g.get_local_range().size())
    target_lid = 0;

  return scratch[target_lid];
}

template <typename T>
__device__ T __acpp_shift_group_left(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return detail::__acpp_shuffle_down_impl(x, delta);
}

// shift_right
template <int Dim, typename T>
__device__ T __acpp_shift_group_right(
    group<Dim> g, T x, typename group<Dim>::linear_id_type delta = 1) {
  __shared__ std::aligned_storage_t<sizeof(T) * detail::max_group_size, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid - delta;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  return scratch[target_lid];
}

template <typename T>
__device__ T __acpp_shift_group_right(
    sub_group g, T x, typename sub_group::linear_id_type delta = 1) {
  return detail::__acpp_shuffle_up_impl(x, delta);
}

// permute_group_by_xor
template <int Dim, typename T>
__device__ T __acpp_permute_group_by_xor(
    group<Dim> g, T x, typename group<Dim>::linear_id_type mask) {
  __shared__ std::aligned_storage_t<sizeof(T) * detail::max_group_size, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  typename group<Dim>::linear_id_type lid        = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid = lid ^ mask;

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  return scratch[target_lid];
}

// permute_group_by_xor
template <typename T>
__device__ T __acpp_permute_group_by_xor(
    sub_group g, T x, typename sub_group::linear_id_type mask) {
  return detail::__acpp_shuffle_xor_impl(x, mask);
}

// select_from_group
template <int Dim, typename T>
__device__ T __acpp_select_from_group(
    group<Dim> g, T x, typename group<Dim>::id_type remote_local_id) {
  __shared__ std::aligned_storage_t<sizeof(T) * detail::max_group_size, alignof(T)>
             scratch_storage;
  T *        scratch = reinterpret_cast<T *>(&scratch_storage);

  typename group<Dim>::linear_id_type lid = g.get_local_linear_id();
  typename group<Dim>::linear_id_type target_lid =
      linear_id<g.dimensions>::get(remote_local_id, g.get_local_range());

  scratch[lid] = x;
  __acpp_group_barrier(g);

  // checking for both larger and smaller in case 'group<Dim>::linear_id_type' is not unsigned
  if (target_lid > g.get_local_range().size() || target_lid < 0)
    target_lid = 0;

  return scratch[target_lid];
}

template <typename T>
__device__ T __acpp_select_from_group(
    sub_group g, T x, typename sub_group::id_type remote_local_id) {
  typename sub_group::linear_id_type target_lid = remote_local_id.get(0);
  return detail::__acpp_shuffle_impl(x, target_lid);
}

} // namespace sycl
} // namespace hipsycl

#endif
#endif // ACPP_LIBKERNEL_DEVICE_GROUP_FUNCTIONS_HPP

