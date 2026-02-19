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
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_metal_symbol_id(const char* s);

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_id.x");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_id.y");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_id.z");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_group_id.x");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_group_id.y");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_group_id.z");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_size.x");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_size.y");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_local_size.z");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_num_groups.x");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_num_groups.y");
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return __acpp_sscp_metal_symbol_id("__acpp_sscp_metal_num_groups.z");
}
