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

#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN uint __acpp_sscp_metal_symbol_simd_id(const char* s);

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id() {
  return __acpp_sscp_metal_symbol_simd_id("__simd_lane_id");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size() {
  const uint sg = __acpp_sscp_get_subgroup_max_size();

  const uint lid_x = __acpp_sscp_get_local_id_x();
  const uint lid_y = __acpp_sscp_get_local_id_y();
  const uint lid_z = __acpp_sscp_get_local_id_z();

  const uint lsz_x = __acpp_sscp_get_local_size_x();
  const uint lsz_y = __acpp_sscp_get_local_size_y();
  const uint lsz_z = __acpp_sscp_get_local_size_z();

  const uint lid = lid_x + lid_y * lsz_x + lid_z * (lsz_x * lsz_y);
  const uint wg  = lsz_x * lsz_y * lsz_z;

  const uint start = (lid / sg) * sg;
  const uint rem = wg - start;

  return (rem < sg) ? rem : sg;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size() {
  return __acpp_sscp_metal_symbol_simd_id("__simd_size");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id() {
  return __acpp_sscp_metal_symbol_simd_id("__simd_group_id");
}
