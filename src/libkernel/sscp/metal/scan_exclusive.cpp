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

#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/scan_inclusive.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"

#include "scan_helpers.hpp"
#include "shuffle_helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_exclusive_scan_##type(__acpp_sscp_algorithm_op op, type value, type unused_init) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_sub_group_exclusive_scan<__acpp_sscp_algorithm_op::plus>(value); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_sub_group_exclusive_scan<__acpp_sscp_algorithm_op::multiply>(value); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}
SCAN_TYPES
#undef X

template<__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_work_group_exclusive_scan(T value, T init) {
  auto* scratch = __acpp_sscp_get_typed_dynamic_local_memory<T>();
  const uint subgroup_size = __acpp_sscp_get_subgroup_max_size();
  const uint lx = __acpp_sscp_get_local_id_x();
  const uint ly = __acpp_sscp_get_local_id_y();
  const uint lz = __acpp_sscp_get_local_id_z();
  const uint tg_x = __acpp_sscp_get_local_size_x();
  const uint tg_y = __acpp_sscp_get_local_size_y();
  const uint tg_z = __acpp_sscp_get_local_size_z();

  const uint lid = (uint)lx + (uint)tg_x * ((uint)ly + (uint)tg_y * (uint)lz);
  const uint local_size = (uint)tg_x * (uint)tg_y * (uint)tg_z;
  const uint group_id = __acpp_sscp_get_subgroup_id();
  const uint lane_id  = __acpp_sscp_get_subgroup_local_id();
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  auto prefix_incl_op = [&](T value) {
    if constexpr(op == __acpp_sscp_algorithm_op::plus) {
      return __acpp_sscp_sub_group_inclusive_scan<__acpp_sscp_algorithm_op::plus>(value);
    } else {
      return __acpp_sscp_sub_group_inclusive_scan<__acpp_sscp_algorithm_op::multiply>(value);
    }
  };

  auto binary_op = [&](T a, T b) {
    if constexpr(op == __acpp_sscp_algorithm_op::plus) {
      return a + b;
    } else {
      return a * b;
    }
  };

  const uint group_base = group_id * subgroup_size;
  uint active = 0;
  if(group_base < local_size) {
    uint rem = local_size - group_base;
    active = (rem < subgroup_size) ? rem : subgroup_size;
  }
  const uint last_lane = (active > 0) ? (active - 1u) : 0u;

  T v = value;
  if(lane_id >= active) v = init;

  const T incl = prefix_incl_op(v);

  T excl = __acpp_sscp_metal_shuffle_up(incl, 1); // lane i gets incl(i-1)
  if(lane_id == 0) excl = init;

  if(lane_id == last_lane) scratch[group_id] = incl;
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  if(ngroups <= subgroup_size) {
    if(lid < ngroups) {
      T s = scratch[lid];
      scratch[lid] = prefix_incl_op(s);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  } else if(ngroups <= 2u * subgroup_size) {
    if(lid < ngroups) {
      T s = scratch[lid];
      scratch[lid] = prefix_incl_op(s);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

    if(lid < ngroups) {
      T add = (lid >= subgroup_size) ? scratch[subgroup_size - 1u] : init;
      scratch[lid] = binary_op(scratch[lid], add);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  } else {
    for(uint offset = 1; offset < ngroups; offset <<= 1) {
      T addend = init;
      if(lid < ngroups && lid >= offset) addend = scratch[lid - offset];

      __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

      if(lid < ngroups && lid >= offset) scratch[lid] = binary_op(scratch[lid], addend);

      __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
    }
    __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  }

  const T group_offset = (group_id > 0) ? scratch[group_id - 1] : init;
  return binary_op(excl, group_offset);
}

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_work_group_exclusive_scan_##type(__acpp_sscp_algorithm_op op, type value, type init) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_work_group_exclusive_scan<__acpp_sscp_algorithm_op::plus>(value, init); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_work_group_exclusive_scan<__acpp_sscp_algorithm_op::multiply>(value, init); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}

SCAN_TYPES
#undef X
