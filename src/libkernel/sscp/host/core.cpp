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
#include <stddef.h>

extern "C" size_t __acpp_cbs_local_id_x;
extern "C" size_t __acpp_cbs_local_id_y;
extern "C" size_t __acpp_cbs_local_id_z;

extern "C" size_t __acpp_cbs_local_size_x;
extern "C" size_t __acpp_cbs_local_size_y;
extern "C" size_t __acpp_cbs_local_size_z;

extern "C" size_t __acpp_cbs_group_id_x;
extern "C" size_t __acpp_cbs_group_id_y;
extern "C" size_t __acpp_cbs_group_id_z;

extern "C" size_t __acpp_cbs_num_groups_x;
extern "C" size_t __acpp_cbs_num_groups_y;
extern "C" size_t __acpp_cbs_num_groups_z;

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return __acpp_cbs_local_id_x;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return __acpp_cbs_local_id_y;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return __acpp_cbs_local_id_z;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return __acpp_cbs_group_id_x;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return __acpp_cbs_group_id_y;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return __acpp_cbs_group_id_z;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return __acpp_cbs_local_size_x;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return __acpp_cbs_local_size_y;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return __acpp_cbs_local_size_z;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return __acpp_cbs_num_groups_x;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return __acpp_cbs_num_groups_y;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return __acpp_cbs_num_groups_z;
}
