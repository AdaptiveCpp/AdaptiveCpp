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

extern "C" __acpp_uint32 __spirv_BuiltInSubgroupLocalInvocationId;
extern "C" __acpp_uint32 __spirv_BuiltInSubgroupSize;
extern "C" __acpp_uint32 __spirv_BuiltInSubgroupMaxSize;
extern "C" __acpp_uint32 __spirv_BuiltInSubgroupId;
extern "C" __acpp_uint32 __spirv_BuiltInNumSubgroups;


HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id() {
  return __spirv_BuiltInSubgroupLocalInvocationId;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size() {
  return __spirv_BuiltInSubgroupSize;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size() {
  return __spirv_BuiltInSubgroupMaxSize;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id() {
  return __spirv_BuiltInSubgroupId;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups() {
  return __spirv_BuiltInNumSubgroups;
}


