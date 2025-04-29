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


#ifndef ACPP_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP
#define ACPP_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../generic/hiplike/warp_shuffle.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA

namespace hipsycl {
namespace sycl::detail::hiplike_builtins {

// barrier
template <int Dim>
__device__ inline void __acpp_group_barrier(group<Dim> g,
                                               memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  }
  __syncthreads();
}

template<int Dim>
__device__
inline void __acpp_group_barrier(group<Dim> g) {
  __syncthreads();
}

__device__ inline void __acpp_group_barrier(sub_group g,
                                               memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
  __syncwarp(); // not necessarily needed, but might improve performance
}

__device__
inline void __acpp_group_barrier(sub_group g) {
  __syncwarp(); // not necessarily needed, but might improve performance
}

// any_of
__device__
inline bool __acpp_any_of_group(sub_group g, bool pred) {
  return __any_sync(detail::AllMask, pred);
}

// all_of
__device__
inline bool __acpp_all_of_group(sub_group g, bool pred) {
  return __all_sync(detail::AllMask, pred);
}

// none_of
__device__
inline bool __acpp_none_of_group(sub_group g, bool pred) {
  return !__any_sync(detail::AllMask, pred);
}

// reduce
template <typename T, typename BinaryOperation>
__device__ T __acpp_reduce_over_group(sub_group g, T x,
                                         BinaryOperation binary_op) {
  const size_t       lid        = g.get_local_linear_id();
  const size_t       lrange     = g.get_local_linear_range();
  const unsigned int activemask = __ballot_sync(0xffffffff, 1);

  auto local_x = x;

  for (size_t i = lrange / 2; i > 0; i /= 2) {
    auto other_x = detail::__acpp_shuffle_impl(local_x, lid + i);
    if (activemask & (1 << (lid + i)))
      local_x = binary_op(local_x, other_x);
  }
  return detail::__acpp_shuffle_impl(local_x, 0);
}

// exclusive_scan
template <typename V, typename T, typename BinaryOperation>
__device__ T __acpp_exclusive_scan_over_group(sub_group g, V x, T init,
                                                 BinaryOperation binary_op) {
  const size_t       lid        = g.get_local_linear_id();
  const size_t       lrange     = g.get_local_linear_range();
  const unsigned int activemask = __ballot_sync(0xffffffff, 1);

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::__acpp_shuffle_impl(local_x, next_id);
    if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  size_t next_id = lid - 1;
  if (g.leader())
    next_id = 0;

  auto return_value = detail::__acpp_shuffle_impl(local_x, next_id);

  if (g.leader())
    return init;

  return binary_op(return_value, init);
}

// inclusive_scan
template <typename T, typename BinaryOperation>
__device__ T __acpp_inclusive_scan_over_group(sub_group g, T x,
                                                 BinaryOperation binary_op) {
  const size_t       lid        = g.get_local_linear_id();
  const size_t       lrange     = g.get_local_linear_range();
  const unsigned int activemask = __ballot_sync(0xffffffff, 1);

  auto local_x = x;

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::__acpp_shuffle_impl(local_x, next_id);
    if (activemask & (1 << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
}

} // namespace sycl
} // namespace hipsycl

#endif
#endif // ACPP_LIBKERNEL_CUDA_GROUP_FUNCTIONS_HPP

