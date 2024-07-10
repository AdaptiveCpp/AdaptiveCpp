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

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id() {
  return __nvvm_read_ptx_sreg_laneid();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size() {

  if (__acpp_sscp_get_subgroup_id() ==
      __acpp_sscp_get_num_subgroups() - 1) {
    auto wg_size = __acpp_sscp_get_local_size_x() *
                   __acpp_sscp_get_local_size_y() *
                   __acpp_sscp_get_local_size_z();

    auto num_max_sized_subgroups = __acpp_sscp_get_num_subgroups() - 1;
    return wg_size -
           num_max_sized_subgroups * __acpp_sscp_get_subgroup_max_size();
  } else {
    return __acpp_sscp_get_subgroup_max_size();
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size() {
  return 32;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id() {
  size_t local_tid =
      __acpp_sscp_get_local_id_x() +
      __acpp_sscp_get_local_id_y() * __acpp_sscp_get_local_size_x() +
      __acpp_sscp_get_local_id_z() * __acpp_sscp_get_local_size_x() *
          __acpp_sscp_get_local_size_y();
  return local_tid / __acpp_sscp_get_subgroup_max_size();
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  auto wg_size = __acpp_sscp_get_local_size_x() *
                 __acpp_sscp_get_local_size_y() *
                 __acpp_sscp_get_local_size_z();
  auto sg_size = __acpp_sscp_get_subgroup_max_size();

  return (wg_size + sg_size - 1) / sg_size;
}
