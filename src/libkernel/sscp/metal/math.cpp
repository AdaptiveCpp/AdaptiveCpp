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

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include <math.h> // NAN

using f32 = float;
using i32 = __acpp_int32;
using i1 = bool;
using u8 = __acpp_uint8;
using u16 = __acpp_uint16;
using u32 = __acpp_uint32;
using u64 = __acpp_uint64;

#define DECLARE_METAL_INLINE_1(ret, t1) \
  HIPSYCL_SSCP_BUILTIN ret __acpp_metal_inline_##ret##_##t1(const char* s, t1 x);

#define DECLARE_METAL_INLINE_2(ret, t1, t2) \
  HIPSYCL_SSCP_BUILTIN ret __acpp_metal_inline_##ret##_##t1##_##t2(const char* s, t1 x, t2 y);

#define DECLARE_METAL_INLINE_3(ret, t1, t2, t3) \
  HIPSYCL_SSCP_BUILTIN ret __acpp_metal_inline_##ret##_##t1##_##t2##_##t3(const char* s, t1 x, t2 y, t3 z);

DECLARE_METAL_INLINE_1(f32, f32)
DECLARE_METAL_INLINE_2(f32, f32, f32)
DECLARE_METAL_INLINE_3(f32, f32, f32, f32)
DECLARE_METAL_INLINE_1(i1, f32)
DECLARE_METAL_INLINE_1(i32, f32)
DECLARE_METAL_INLINE_2(f32, f32, i32)

DECLARE_METAL_INLINE_1(u8, u8)
DECLARE_METAL_INLINE_1(u16, u16)
DECLARE_METAL_INLINE_1(u32, u32)
DECLARE_METAL_INLINE_1(u64, u64)

DECLARE_METAL_INLINE_2(u16, u16, u16)

#define METAL_INLINE_1(ret, t1) __acpp_metal_inline_##ret##_##t1
#define METAL_INLINE_2(ret, t1, t2) __acpp_metal_inline_##ret##_##t1##_##t2
#define METAL_INLINE_3(ret, t1, t2, t3) __acpp_metal_inline_##ret##_##t1##_##t2##_##t3

#define HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x) { \
    return METAL_INLINE_1(f32, f32)(#name, x); \
  }

#define HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x, f32 y) { \
    return METAL_INLINE_2(f32, f32, f32)(#name, x, y); \
  }

#define HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN3(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x, f32 y, f32 z) { \
    return METAL_INLINE_3(f32, f32, f32, f32)(#name, x, y, z); \
  }

HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(tan)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(asin)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(acos)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(atan)
// warning: atan2(0, 0) is nan in metal, but in C:
// atan2(±0, −0) returns ±π
// atan2(±0, +0) returns ±0.
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(atan2)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(sinh)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(cosh)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(tanh)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(cos)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(sin)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(exp)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(exp2)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(exp10)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(log)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(log2)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(log10)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(sqrt)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(rsqrt)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(floor)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(ceil)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(round)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(trunc)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(rint)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN(fabs)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(copysign)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN3(fma)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmin)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmax)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmod)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(fdim)
HIPSYCL_SSCP_MAP_METAL_FLOAT_BUILTIN2(pow)

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isnan_f32(f32 x) {
  return METAL_INLINE_1(i32, f32)("isnan", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isinf_f32(f32 x) {
  return METAL_INLINE_1(i32, f32)("isinf", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isfinite_f32(f32 x) {
  return METAL_INLINE_1(i32, f32)("isfinite", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isnormal_f32(f32 x) {
  return METAL_INLINE_1(i32, f32)("isnormal", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_signbit_f32(f32 x) {
  return METAL_INLINE_1(i32, f32)("signbit", x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_ldexp_f32(f32 x, i32 k) {
  return METAL_INLINE_2(f32, f32, i32)("ldexp", x, k);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_mad_f32(f32 x, f32 y, f32 z) {
  return __acpp_sscp_fma_f32(x, y, z);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_log1p_f32(f32 x) {
  f32 u = 1.0f + x;
  if (u == 1.0f) return x;
  return __acpp_sscp_log_f32(u) * x / (u - 1.0f);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_expm1_f32(f32 x) {
  f32 u = __acpp_sscp_exp_f32(x);
  if (u == 1.0f) return x;
  if (u - 1.0f == -1.0f) return -1.0f;
  return (u - 1.0f) * x / __acpp_sscp_log_f32(u);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_rootn_f32(f32 x, i32 n) {
  if (n == 0) return (f32)NAN;

  if (x < 0.0f) {
      if ((n & 1) == 0) {
          return (f32)NAN;
      }
      return -__acpp_sscp_pow_f32(-x, 1.0f / f32(n));
  }

  return __acpp_sscp_pow_f32(x, 1.0f / f32(n));
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_pown_f32(f32 x, i32 n) {
  return METAL_INLINE_2(f32, f32, i32)("pow(%s, as_type<int>(%s))", x, n);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_powr_f32(f32 x, f32 n) {
  return METAL_INLINE_2(f32, f32, f32)("pow", x, n);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_hypot_f32(f32 x, f32 y) {
  return METAL_INLINE_2(f32, f32, f32)("length(float2(%s, %s))", x, y);
}

#define HIPSYCL_SSCP_MAP_METAL_BITOP(op, type, suffix) \
  HIPSYCL_SSCP_BUILTIN type __acpp_sscp_##op##_##suffix(type x) { \
    return METAL_INLINE_1(suffix, suffix)(#op, x); \
  }

HIPSYCL_SSCP_MAP_METAL_BITOP(ctz, u8, u8)
HIPSYCL_SSCP_MAP_METAL_BITOP(ctz, u16, u16)
HIPSYCL_SSCP_MAP_METAL_BITOP(ctz, u32, u32)
HIPSYCL_SSCP_MAP_METAL_BITOP(ctz, u64, u64)

HIPSYCL_SSCP_MAP_METAL_BITOP(clz, u8, u8)
HIPSYCL_SSCP_MAP_METAL_BITOP(clz, u16, u16)
HIPSYCL_SSCP_MAP_METAL_BITOP(clz, u32, u32)
HIPSYCL_SSCP_MAP_METAL_BITOP(clz, u64, u64)

HIPSYCL_SSCP_MAP_METAL_BITOP(popcount, u8, u8)
HIPSYCL_SSCP_MAP_METAL_BITOP(popcount, u16, u16)
HIPSYCL_SSCP_MAP_METAL_BITOP(popcount, u32, u32)
HIPSYCL_SSCP_MAP_METAL_BITOP(popcount, u64, u64)

#define HIPSYCL_SSCP_MAP_METAL_HALF_BINOP(op, symbol) \
  HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_half_##op(u16 x, u16 y) { \
    return METAL_INLINE_2(u16, u16, u16)("as_type<ushort>(as_type<half>(%s) " #symbol " as_type<half>(%s))", x, y); \
  }

HIPSYCL_SSCP_MAP_METAL_HALF_BINOP(add, +)
HIPSYCL_SSCP_MAP_METAL_HALF_BINOP(sub, -)
HIPSYCL_SSCP_MAP_METAL_HALF_BINOP(mul, *)
HIPSYCL_SSCP_MAP_METAL_HALF_BINOP(div, /)

