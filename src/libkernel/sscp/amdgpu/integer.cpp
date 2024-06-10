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
