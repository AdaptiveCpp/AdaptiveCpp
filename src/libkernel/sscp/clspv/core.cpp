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

__acpp_uint64 get_local_id(unsigned);
__acpp_uint64 get_group_id(unsigned);
__acpp_uint64 get_local_size(unsigned);
__acpp_uint64 get_num_groups(unsigned);

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return get_local_id(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return get_local_id(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return get_local_id(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return get_group_id(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return get_group_id(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return get_group_id(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return get_local_size(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return get_local_size(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return get_local_size(2);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return get_num_groups(0);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return get_num_groups(1);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return get_num_groups(2);
}
