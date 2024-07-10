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

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_x() {
  return __nvvm_read_ptx_sreg_tid_x();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_y() {
  return __nvvm_read_ptx_sreg_tid_y();;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_id_z() {
  return __nvvm_read_ptx_sreg_tid_z();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_x() {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_y() {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_group_id_z() {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_local_size_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_get_num_groups_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}
