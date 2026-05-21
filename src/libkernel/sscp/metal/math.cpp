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
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"

#include <math.h> // NAN

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32(const char* name, f32 x);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_f32(const char* name, f32 x, f32 y);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_f32_f32(const char* name, f32 x, f32 y, f32 z);

#define ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x) { \
    return __acpp_sscp_metal_math_f32_f32(#name, x); \
  }

#define ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x, f32 y) { \
    return __acpp_sscp_metal_math_f32_f32_f32(#name, x, y); \
  }

#define ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN3(name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_##name##_f32(f32 x, f32 y, f32 z) { \
    return __acpp_sscp_metal_math_f32_f32_f32_f32(#name, x, y, z); \
  }

ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(tan)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(asin)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(acos)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(atan)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(atan2)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(sinh)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(cosh)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(tanh)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(tanpi)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(cos)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(cospi)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(sin)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(sinpi)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(exp)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(exp2)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(exp10)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(log)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(log2)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(log10)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(sqrt)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(rsqrt)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(floor)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(ceil)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(round)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(trunc)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(rint)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(fabs)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(copysign)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN3(fma)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmin)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmax)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(fmod)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(fdim)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(pow)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN2(nextafter)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(acosh)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(asinh)
ACPP_SSCP_MAP_METAL_FLOAT_BUILTIN(atanh)

// See: https://en.wikipedia.org/wiki/Gamma_function  (Log-gamma function)
// log Gamma(z) = (z - 1/2) log(z) - z + 1/2 log(2pi) + 1/(12z) - 1/(360 z^3) + 1/(1260 z^5) + ....
// log Gamma(z-m) = log Gamma(z) - sum_{k=1}^{m} log(z-k)
// threshold: 1/(1260*(8^5)) ~ 2.4e-8
static f32 lgamma_positive_f32(f32 z) {
  static constexpr f32 log_2pi_2 = 0.9189385332046727f; // = 1/2 log(2pi)
  f32 adj = 0.0f;
  while (z < 8.0f) {
    adj += __acpp_sscp_log_f32(z);
    z += 1.0f;
  }
  f32 z2 = z*z;
  f32 z3 = z2*z;

  f32 lg = log_2pi_2 + (z - 0.5f) * __acpp_sscp_log_f32(z) - z
           + 1.0f / (12.0f * z)
           - 1.0f / (360.0f * z3);
  return lg - adj;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_lgamma_f32(f32 x) {
  if (x > 0.0f) {
    return lgamma_positive_f32(x);
  }
  // reflection
  return __acpp_sscp_log_f32((f32)M_PI / __acpp_sscp_fabs_f32(__acpp_sscp_sinpi_f32(x))) - lgamma_positive_f32(1.0f - x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_cbrt_f32(f32 x) {
  if (x == 0.0f) return 0.0f;
  f32 ax = __acpp_sscp_fabs_f32(x);
  f32 result = __acpp_sscp_pow_f32(ax, 1.0f / 3.0f);
  return (x < 0.0f) ? -result : result;
}

// erf: Abramowitz & Stegun 7.1.26, max error 1.5e-7, see https://en.wikipedia.org/wiki/Error_function
inline f32 __acpp_sscp_erfc_poly_f32(f32 x) {
  f32 p = 0.3275911f;
  f32 a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f;
  f32 t = 1.0f / (1.0f + p * x);
  return (((((a5*t + a4)*t + a3)*t) + a2)*t + a1)*t * __acpp_sscp_exp_f32(-x*x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_erf_f32(f32 x) {
  if (x < 0.0f) {
    return __acpp_sscp_erfc_poly_f32(-x) - 1.0f;
  } else {
    return 1.0f - __acpp_sscp_erfc_poly_f32(x);
  }
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_erfc_f32(f32 x) {
  if (x < 0.0f) {
    return 2.0f - __acpp_sscp_erfc_poly_f32(-x);
  } else {
    return __acpp_sscp_erfc_poly_f32(x);
  }
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_tgamma_f32(f32 x) {
  if (x > 0.0f) {
    return __acpp_sscp_exp_f32(__acpp_sscp_lgamma_f32(x));
  }
  // reflection
  return (f32)M_PI / (__acpp_sscp_sinpi_f32(x) * __acpp_sscp_exp_f32(__acpp_sscp_lgamma_f32(1.0f - x)));
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_remainder_f32(f32 x, f32 y) {
  return x - __acpp_sscp_rint_f32(x / y) * y;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_acospi_f32(f32 x) {
  return __acpp_sscp_acos_f32(x) / (f32)M_PI;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_asinpi_f32(f32 x) {
  return __acpp_sscp_asin_f32(x) / (f32)M_PI;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atanpi_f32(f32 x) {
  return __acpp_sscp_atan_f32(x) / (f32)M_PI;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atan2pi_f32(f32 x, f32 y) {
  return __acpp_sscp_atan2_f32(x, y) / (f32)M_PI;
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_metal_math_i32_f32(const char* name, f32 x);

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_ilogb_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("ilogb", __acpp_sscp_fabs_f32(x));
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_logb_f32(f32 x) {
  return (f32)__acpp_sscp_metal_math_i32_f32("ilogb", __acpp_sscp_fabs_f32(x));
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_maxmag_f32(f32 x, f32 y) {
  f32 ax = __acpp_sscp_fabs_f32(x);
  f32 ay = __acpp_sscp_fabs_f32(y);
  if (ax == ay) return __acpp_sscp_fmax_f32(x, y);
  return (ax > ay) ? x : y;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_minmag_f32(f32 x, f32 y) {
  f32 ax = __acpp_sscp_fabs_f32(x);
  f32 ay = __acpp_sscp_fabs_f32(y);
  if (ax == ay) return __acpp_sscp_fmin_f32(x, y);
  return (ax < ay) ? x : y;
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_fract_f32(f32 x, f32* iptr) {
  *iptr = __acpp_sscp_floor_f32(x);
  return __acpp_sscp_fmin_f32(x - *iptr, __acpp_sscp_nextafter_f32(1.0f, 0.0f));
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_i32ptr(const char* name, f32 x, i32* p);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_f32ptr(const char* name, f32 x, f32* p);

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_frexp_f32(f32 x, i32* exp) {
  return __acpp_sscp_metal_math_f32_f32_i32ptr("frexp(%s, *__pointer_cast<int>(%s))", x, exp);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_modf_f32(f32 x, f32* iptr) {
  return __acpp_sscp_metal_math_f32_f32_f32ptr("modf(%s, *__pointer_cast<float>(%s))", x, iptr);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_lgamma_r_f32(f32 x, i32* signp) {
  f32 r = __acpp_sscp_lgamma_f32(x);
  if (x >= 0.0f) {
    *signp = 1;
  } else {
    i32 n = (i32)__acpp_sscp_floor_f32(-x);
    *signp = (n & 1) ? 1 : -1;
  }
  return r;
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isnan_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("isnan", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isinf_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("isinf", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isfinite_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("isfinite", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_isnormal_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("isnormal", x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_signbit_f32(f32 x) {
  return __acpp_sscp_metal_math_i32_f32("signbit", x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_ldexp_f32(f32 x, i32 k) {
  return __acpp_sscp_metal_math_f32_f32_f32("ldexp", x, k);
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

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_i32(const char* name, f32 x, i32 n);

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_pown_f32(f32 x, i32 n) {
  return __acpp_sscp_metal_math_f32_f32_i32("pow(%s, as_type<int>(%s))", x, n);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_powr_f32(f32 x, f32 n) {
  return __acpp_sscp_metal_math_f32_f32_f32("pow", x, n);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_hypot_f32(f32 x, f32 y) {
  return __acpp_sscp_metal_math_f32_f32_f32("length(float2(%s, %s))", x, y);
}

#define BITOP_TYPES \
  X(u8) \
  X(u16) \
  X(u32) \
  X(u64)

#define X(type) HIPSYCL_SSCP_BUILTIN type __acpp_sscp_metal_bitop_##type(const char* name, type x);
BITOP_TYPES
#undef X

#define ACPP_SSCP_MAP_METAL_BITOP(op, type) \
  HIPSYCL_SSCP_BUILTIN type __acpp_sscp_##op##_##type(type x) { \
    return __acpp_sscp_metal_bitop_##type(#op, x); \
  }

ACPP_SSCP_MAP_METAL_BITOP(ctz, u8)
ACPP_SSCP_MAP_METAL_BITOP(ctz, u16)
ACPP_SSCP_MAP_METAL_BITOP(ctz, u32)
ACPP_SSCP_MAP_METAL_BITOP(ctz, u64)

ACPP_SSCP_MAP_METAL_BITOP(clz, u8)
ACPP_SSCP_MAP_METAL_BITOP(clz, u16)
ACPP_SSCP_MAP_METAL_BITOP(clz, u32)
ACPP_SSCP_MAP_METAL_BITOP(clz, u64)

ACPP_SSCP_MAP_METAL_BITOP(popcount, u8)
ACPP_SSCP_MAP_METAL_BITOP(popcount, u16)
ACPP_SSCP_MAP_METAL_BITOP(popcount, u32)
ACPP_SSCP_MAP_METAL_BITOP(popcount, u64)

HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_metal_half_op(const char* name, u16 x, u16 y);

#define ACPP_SSCP_MAP_METAL_HALF_BINOP(op, symbol) \
  HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_half_##op(u16 x, u16 y) { \
    return __acpp_sscp_metal_half_op("as_type<ushort>(as_type<half>(%s) " #symbol " as_type<half>(%s))", x, y); \
  }

ACPP_SSCP_MAP_METAL_HALF_BINOP(add, +)
ACPP_SSCP_MAP_METAL_HALF_BINOP(sub, -)
ACPP_SSCP_MAP_METAL_HALF_BINOP(mul, *)
ACPP_SSCP_MAP_METAL_HALF_BINOP(div, /)
