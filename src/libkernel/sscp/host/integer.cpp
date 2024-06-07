/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


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
