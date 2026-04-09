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
#include "hipSYCL/sycl/libkernel/sscp/builtins/collpredicate.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_collpredicate(const char* s, bool pred);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred) {
  return __acpp_sscp_metal_collpredicate("simd_any", pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred) {
  return __acpp_sscp_metal_collpredicate("simd_all", pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred) {
  return !__acpp_sscp_metal_collpredicate("simd_any", pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool predicate) {
  __attribute__((address_space(3))) bool* scratch = __acpp_sscp_get_typed_dynamic_local_memory<bool>();

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
  const uint subgroup_size = __acpp_sscp_get_subgroup_max_size();
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  bool sg_any = __acpp_sscp_metal_collpredicate("simd_any", predicate);
  if (lane_id == 0) {
    scratch[group_id] = sg_any;
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  if (group_id == 0) {
    bool v = false;
    if (lid < ngroups) {
      v = scratch[lid];
    }
    bool wg_any = __acpp_sscp_metal_collpredicate("simd_any", v);
    if (lid == 0) {
      scratch[0] = wg_any;
    }
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  return scratch[0];
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool predicate) {
  __attribute__((address_space(3))) bool* scratch = __acpp_sscp_get_typed_dynamic_local_memory<bool>();

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
  const uint subgroup_size = __acpp_sscp_get_subgroup_max_size();
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  bool sg_all = __acpp_sscp_metal_collpredicate("simd_all", predicate);
  if (lane_id == 0) {
    scratch[group_id] = sg_all;
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  if (group_id == 0) {
    bool v = true;
    if (lid < ngroups) {
      v = scratch[lid];
    }
    bool wg_all = __acpp_sscp_metal_collpredicate("simd_all", v);
    if (lid == 0) {
      scratch[0] = wg_all;
    }
  }
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  return scratch[0];
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool predicate) {
  return !__acpp_sscp_work_group_any(predicate);
}
