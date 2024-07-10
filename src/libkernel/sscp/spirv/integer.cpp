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
#include "hipSYCL/sycl/libkernel/sscp/builtins/integer.hpp"

__acpp_int32 __spirv_ocl_s_mul24(__acpp_int32 a, __acpp_int32 b);
__acpp_uint32 __spirv_ocl_u_mul24(__acpp_uint32 a, __acpp_uint32 b);

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_mul24_s32(__acpp_int32 a, __acpp_int32 b) {
  return __spirv_ocl_s_mul24(a, b);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_mul24_u32(__acpp_uint32 a, __acpp_uint32 b) {
  return __spirv_ocl_u_mul24(a, b);
}


__acpp_int32 __spirv_ocl_clz(__acpp_int32 a);
__acpp_int64 __spirv_ocl_clz(__acpp_int64 a);
__acpp_uint32 __spirv_ocl_clz(__acpp_uint32 a);
__acpp_uint64 __spirv_ocl_clz(__acpp_uint64 a);


HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_clz_u32(__acpp_uint32 a){
  return __spirv_ocl_clz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_clz_u64(__acpp_uint64 a){
  return __spirv_ocl_clz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_clz_u8(__acpp_uint8 a){
  return __acpp_sscp_clz_u32(a)-24;
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_clz_u16(__acpp_uint16 a){
  return __acpp_sscp_clz_u32(a)-16;
}

__acpp_uint32 __spirv_ocl_popcount(__acpp_uint32 a);
__acpp_uint64 __spirv_ocl_popcount(__acpp_uint64 a);

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_popcount_u32(__acpp_uint32 a){
  return __spirv_ocl_popcount(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_popcount_u64(__acpp_uint64 a){
  return __spirv_ocl_popcount(a);
}
