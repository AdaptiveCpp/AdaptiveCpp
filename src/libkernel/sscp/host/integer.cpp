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

#include <limits.h>

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_mul24_s32(__acpp_int32 a, __acpp_int32 b) {
  __builtin_assume(a >= -8388608 && a <= 8388607);
  __builtin_assume(b >= -8388608 && b <= 8388607);

  return a * b;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_mul24_u32(__acpp_uint32 a, __acpp_uint32 b) {
  __builtin_assume(a >= 0 && a <= 16777215);
  __builtin_assume(b >= 0 && b <= 16777215);
  return a * b;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_clz_u8(__acpp_uint8 a){
  // builtin_clz(0) is UB on some arch
  if (a == 0) {
    return CHAR_BIT;
  }

  constexpr __acpp_uint8 diff = CHAR_BIT*(sizeof(unsigned int) - sizeof(__acpp_uint8));
  return __builtin_clz(a) - diff;
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_clz_u16(__acpp_uint16 a){
  // builtin_clz(0) is UB on some arch
  if (a == 0) {
    return CHAR_BIT*sizeof(__acpp_uint16);
  }

  constexpr __acpp_uint16 diff = CHAR_BIT*(sizeof(unsigned int) - sizeof(__acpp_uint16));
  return __builtin_clz(a) - diff;
}
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_clz_u32(__acpp_uint32 a){
  // builtin_clz(0) is UB on some arch
  if (a == 0) {
    return CHAR_BIT*sizeof(__acpp_uint32);
  }
  return __builtin_clz(a);
}	
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_clz_u64(__acpp_uint64 a){
  // builtin_clz(0) is UB on some arch
  if (a == 0) {
    return CHAR_BIT*sizeof(__acpp_uint64);
  }
  return __builtin_clzll(a);
}


HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_popcount_u8(__acpp_uint8 a){
  return __builtin_popcount(a & 0xff);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_popcount_u16(__acpp_uint16 a){
  return __builtin_popcount(a & 0xffff);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_popcount_u32(__acpp_uint32 a){
  return __builtin_popcount(a);
}	
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_popcount_u64(__acpp_uint64 a){
  return __builtin_popcountll(a);
}
