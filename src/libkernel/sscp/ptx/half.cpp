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
#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"

using hipsycl::fp16::as_integer;

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_add(__acpp_f16 a,
                                                     __acpp_f16 b) {
  __acpp_uint16 result;
  asm("{add.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_sub(__acpp_f16 a,
                                                     __acpp_f16 b) {
  __acpp_uint16 result;
  asm("{sub.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_mul(__acpp_f16 a,
                                                     __acpp_f16 b) {
  __acpp_uint16 result;
  asm("{mul.f16 %0,%1,%2;\n}"
    : "=h"(result)
    : "h"(as_integer(a)),"h"(as_integer(b)));
  return hipsycl::fp16::create(result);
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_div(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(__nv_fast_fdividef(
      hipsycl::fp16::promote_to_float(a), hipsycl::fp16::promote_to_float(b)));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lt(__acpp_f16 a, __acpp_f16 b) {
  __acpp_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.lt.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lte(__acpp_f16 a, __acpp_f16 b) {
  __acpp_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.le.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gt(__acpp_f16 a, __acpp_f16 b) {
  __acpp_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.gt.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gte(__acpp_f16 a, __acpp_f16 b) {
  __acpp_uint16 v;
  asm( "{ .reg .pred __$temp3;\n"
      "  setp.ge.f16  __$temp3, %1, %2;\n"
      "  selp.u16 %0, 1, 0, __$temp3;}"
      : "=h"(v) : "h"(as_integer(a)), "h"(as_integer(b)));
  return v != 0;
}
