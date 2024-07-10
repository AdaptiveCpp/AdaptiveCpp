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
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu/ockl.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/integer.hpp"


HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_mul24_s32(__acpp_int32 a, __acpp_int32 b) {
  return __ockl_mul24_i32(a, b);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_mul24_u32(__acpp_uint32 a, __acpp_uint32 b) {
  return __ockl_mul24_u32(a, b);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_clz_u8(__acpp_uint8 a){
  return __ockl_clz_u8(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_clz_u16(__acpp_uint16 a){
  return __ockl_clz_u16(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_clz_u32(__acpp_uint32 a){
  return __ockl_clz_u32(a);
}	
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_clz_u64(__acpp_uint64 a){
  return __ockl_clz_u64(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_popcount_u32(__acpp_uint32 a){
  return __ockl_popcount_u32(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_popcount_u64(__acpp_uint64 a){
  return __ockl_popcount_u64(a);
}
