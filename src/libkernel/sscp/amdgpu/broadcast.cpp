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

#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/broadcast.hpp"

// llvm.amdgcn.readlane(src, lane): reads a value from a specific lane index.
// Note: this is distinct from llvm.amdgcn.readfirstlane (no lane argument),
// which reads from the first active lane — used in adaptivecpp::amdgpu::__acpp_sscp_amdgcn_readfirstlane.
extern "C" __acpp_int32 __acpp_sscp_custom_intrinsic__llvm_amdgcn_readlane(
    __acpp_int32 src, __acpp_int32 lane);

template <typename T>
T readlane_helper(T x, __acpp_int32 sender) {
  if constexpr (sizeof(T) <= 4) {
    return static_cast<T>(__acpp_sscp_custom_intrinsic__llvm_amdgcn_readlane(
        static_cast<__acpp_int32>(x), sender));
  } else {
    __acpp_int32 parts[2];
    __builtin_memcpy(&parts[0], &x, 8);
    parts[0] = __acpp_sscp_custom_intrinsic__llvm_amdgcn_readlane(parts[0], sender);
    parts[1] = __acpp_sscp_custom_intrinsic__llvm_amdgcn_readlane(parts[1], sender);
    T result;
    __builtin_memcpy(&result, &parts[0], 8);
    return result;
  }
}

#define ACPP_SUBGROUP_BCAST(fn_suffix, input_type)                                                 \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##input_type __acpp_sscp_sub_group_broadcast_##fn_suffix(__acpp_int32 sender,             \
                                                                  __acpp_##input_type x) {         \
    return readlane_helper(x, sender);                                                             \
  }

#define ACPP_WORKGROUP_BCAST(fn_suffix, input_type)                                                \
  HIPSYCL_SSCP_CONVERGENT_BUILTIN                                                                  \
  __acpp_##input_type __acpp_sscp_work_group_broadcast_##fn_suffix(__acpp_int32 sender,            \
                                                                   __acpp_##input_type x) {        \
    ACPP_SHMEM_ATTRIBUTE __acpp_##input_type shrd_x[1];                                            \
    return hipsycl::libkernel::sscp::wg_broadcast(sender, x, &shrd_x[0]);                          \
  }

ACPP_WORKGROUP_BCAST(i8, int8)
ACPP_WORKGROUP_BCAST(i16, int16)
ACPP_WORKGROUP_BCAST(i32, int32)
ACPP_WORKGROUP_BCAST(i64, int64)

ACPP_SUBGROUP_BCAST(i8, int8)
ACPP_SUBGROUP_BCAST(i16, int16)
ACPP_SUBGROUP_BCAST(i32, int32)
ACPP_SUBGROUP_BCAST(i64, int64)