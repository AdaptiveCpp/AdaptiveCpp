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

#ifndef ACPP_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP
#define ACPP_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

#include "../backend.hpp"
#include "../generic/hiplike/warp_shuffle.hpp"
#include "../id.hpp"
#include "../sub_group.hpp"
#include "../vec.hpp"
#include <type_traits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl::detail::hiplike_builtins {

// barrier
template<int Dim>
__device__
inline void __acpp_group_barrier(group<Dim> g, memory_scope fence_scope) {
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

__device__
inline void __acpp_group_barrier(sub_group g, memory_scope fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
  // threads run in lock-step no sync needed
}
__device__
inline void __acpp_group_barrier(sub_group g) {
}

// any_of
__device__
inline bool __acpp_any_of_group(sub_group g, bool pred) {
  return __any(pred);
}

// all_of
__device__
inline bool __acpp_all_of_group(sub_group g, bool pred) {
  return __all(pred);
}

// none_of
__device__
inline bool __acpp_none_of_group(sub_group g, bool pred) {
  return !__any(pred);
}

// reduce
template <typename T, typename BinaryOperation>
__device__ T __acpp_reduce_over_group(sub_group g, T x,
                                         BinaryOperation binary_op) {
  auto     local_x = x;
  uint64_t activemask = __ballot(1);

  auto lid = g.get_local_linear_id();

  size_t lrange = g.get_local_range().size();

  __acpp_group_barrier(g);

  for (size_t i = lrange / 2; i > 0; i /= 2) {
    auto other_x = detail::__acpp_shuffle_impl(local_x, lid + i);

    // check if target thread exists/is active
    if (activemask & (1l << (lid + i)))
      local_x = binary_op(local_x, other_x);
  }
  return detail::__acpp_shuffle_impl(local_x, 0);
}

// inclusive_scan
template <typename T, typename BinaryOperation>
__device__ T __acpp_inclusive_scan_over_group(sub_group g, T x,
                                                 BinaryOperation binary_op) {
  auto         local_x = x;
  const size_t lid     = g.get_local_linear_id();

  uint64_t activemask = __ballot(1);

  size_t lrange = g.get_local_linear_range();

  for (size_t i = 1; i < lrange; i *= 2) {
    size_t next_id = lid - i;
    if (i > lid)
      next_id = 0;

    auto other_x = detail::__acpp_shuffle_impl(local_x, next_id);
    if (activemask & (1l << (next_id)) && i <= lid && lid < lrange)
      local_x = binary_op(local_x, other_x);
  }

  return local_x;
}

// exclusive_scan
template <typename V, typename T, typename BinaryOperation>
__device__ T __acpp_exclusive_scan_over_group(sub_group g, V x, T init,
                                                 BinaryOperation binary_op) {
  const size_t lid     = g.get_local_linear_id();
  auto         local_x = x;

  local_x = detail::__acpp_shuffle_up_impl(local_x, 1);
  if (lid == 0)
    local_x = init;

  return __acpp_inclusive_scan_over_group(g, local_x, binary_op);
}

} // namespace sycl
} // namespace hipsycl

#endif 
#endif // ACPP_LIBKERNEL_HIP_GROUP_FUNCTIONS_HPP

