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

 #include "hipSYCL/sycl/libkernel/sscp/builtins/collpredicate.hpp"
 #include "hipSYCL/sycl/libkernel/sscp/builtins/spirv/spirv_common.hpp"

bool __spirv_GroupAny(__spv::ScopeFlag scope, bool pred) noexcept;
bool __spirv_GroupAll(__spv::ScopeFlag scope, bool pred) noexcept;

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool pred){
    return __spirv_GroupAny(__spv::ScopeFlag::Workgroup, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred){
    return __spirv_GroupAll(__spv::ScopeFlag::Workgroup, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred){
    return !__spirv_GroupAny(__spv::ScopeFlag::Workgroup, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred){
    return __spirv_GroupAny(__spv::ScopeFlag::Subgroup, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred){
    return __spirv_GroupAll(__spv::ScopeFlag::Subgroup, pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred){
    return !__spirv_GroupAny(__spv::ScopeFlag::Subgroup, pred);
}
