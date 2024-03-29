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

#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"

using hipsycl::fp16::as_integer;

HIPSYCL_SSCP_BUILTIN __hipsycl_f16
__hipsycl_sscp_half_add(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  __hipsycl_uint16 result;
  asm("{add.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f16
__hipsycl_sscp_half_sub(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  __hipsycl_uint16 result;
  asm("{sub.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f16
__hipsycl_sscp_half_mul(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  __hipsycl_uint16 result;
  asm("{mul.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f16
__hipsycl_sscp_half_div(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  return hipsycl::fp16::create(__nv_fast_fdividef(
      hipsycl::fp16::promote_to_float(a), hipsycl::fp16::promote_to_float(b)));
}


HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_lt(__hipsycl_f16 a,
                       __hipsycl_f16 b) {
  __hipsycl_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.lt.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_lte(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  __hipsycl_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.le.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_gt(__hipsycl_f16 a,
                       __hipsycl_f16 b) {
  __hipsycl_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.gt.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_gte(__hipsycl_f16 a,
                        __hipsycl_f16 b) {
  __hipsycl_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.ge.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}
