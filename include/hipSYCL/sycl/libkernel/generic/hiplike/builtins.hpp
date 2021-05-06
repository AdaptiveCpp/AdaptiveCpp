/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_LIBKERNEL_HIPLIKE_BUILTINS_HPP
#define HIPSYCL_LIBKERNEL_HIPLIKE_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"

#include <cstdlib>
#include <cmath>
#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl {
namespace detail {

// ********************** math builtins *********************

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(name, impl_name_sp, impl_name_dp)  \
  HIPSYCL_BUILTIN float name(float x) { return ::impl_name_sp(x); }            \
  HIPSYCL_BUILTIN double name(double x) { return ::impl_name_dp(x); }

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(name, impl_name_sp, impl_name_dp) \
  HIPSYCL_BUILTIN float name(float x, float y) {                               \
    return ::impl_name_sp(x, y);                                               \
  }                                                                            \
  HIPSYCL_BUILTIN double name(double x, double y) {                            \
    return ::impl_name_dp(x, y);                                               \
  }

#define HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN3(name, impl_name_sp, impl_name_dp) \
  HIPSYCL_BUILTIN float name(float x, float y, float z) {                      \
    return ::impl_name_sp(x, y, z);                                            \
  }                                                                            \
  HIPSYCL_BUILTIN double name(double x, double y, double z) {                  \
    return ::impl_name_dp(x, y, z);                                            \
  }

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_acos, acosf, acos)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_acosh, acoshf, acosh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acospi(T x) noexcept {
  return __hipsycl_acos(x) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_asin, asinf, asin)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_asinh, asinhf, asinh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinpi(T x) noexcept {
  return __hipsycl_asin(x) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_atan, atanf, atan)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_atan2, atan2f, atan2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_atanh, atanhf, atanh)


template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanpi(T x) noexcept {
  return __hipsycl_atan(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2pi(T x, T y) noexcept {
  return __hipsycl_atan2(x, y) / M_PI;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_cbrt, cbrtf, cbrt)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_ceil, ceilf, ceil)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_copysign, copysignf, copysign)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_cos, cosf, cos)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_cosh, coshf, cosh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cospi(T x) noexcept {
  return __hipsycl_cos(x * M_PI);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_erf, erff, erf)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_erfc, erfcf, erfc)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_exp, expf, exp)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_exp2, exp2f, exp2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_pow, powf, pow)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp10(T x) noexcept {
  return __hipsycl_pow(static_cast<T>(10), x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_expm1, expm1f, expm1)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_fabs, fabsf, fabs)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_fdim, fdimf, fdim)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_floor, floorf, floor)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN3(__hipsycl_fma, fmaf, fma)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_fmax, fmaxf, fmax)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_fmin, fminf, fmin)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_fmod, fmodf, fmod)

// Unsupported
template<class T>
T __hipsycl_fract(T x, T* ptr) noexcept;

// Unsupported
template<class T, class IntPtr>
T __hipsycl_frexp(T x, IntPtr y) noexcept;

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_hypot, hypotf, hypot)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_ilogb, ilogbf, ilogb)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_lgamma, lgammaf, lgamma)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_tgamma, tgammaf, tgamma)

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_lgamma_r(T x, IntPtr y) noexcept {
  auto r = __hipsycl_lgamma(x);
  auto g = __hipsycl_tgamma(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_log, logf, log)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_log2, log2f, log2)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_log10, log10f, log10)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_log1p, log1pf, log1p)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_logb, logbf, logb)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mad(T x, T y, T z) noexcept {
  return __hipsycl_fma(x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_maxmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return __hipsycl_fmax(x,y);
  return (abs_x > abs_y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_minmag(T x, T y) noexcept {
  auto abs_x = (x < 0) ? -x : x;
  auto abs_y = (y < 0) ? -y : y;
  if(abs_x == abs_y) return __hipsycl_fmin(x,y);
  return (abs_x < abs_y) ? x : y;
}

// Not yet supported
template<class T, class FloatPtr>
T __hipsycl_modf(T x, FloatPtr y) noexcept;

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_nextafter, nextafterf, nextafter)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_powr(T x, T y) noexcept {
  return __hipsycl_pow(x, y);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_pown(T x, IntType y) noexcept {
  return __hipsycl_pow(x, static_cast<T>(y));
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN2(__hipsycl_remainder, remainderf, remainder)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_rint, rintf, rint)

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_rootn(T x, IntType y) noexcept {
  return __hipsycl_pow(x, T{1}/T{y});
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_round, roundf, round)

HIPSYCL_BUILTIN float __hipsycl_rsqrt(float x) noexcept {
  return __frsqrt_rn(x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_sqrt, sqrtf, sqrt)

HIPSYCL_BUILTIN double __hipsycl_rsqrt(double x) noexcept {
  return 1. / __hipsycl_sqrt(x);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_sin, sinf, sin)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sinpi(T x) noexcept {
  return __hipsycl_sin(x * M_PI);
}

HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_tan, tanf, tan)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_tanh, tanhf, tanh)
HIPSYCL_DEFINE_HIPLIKE_MATH_BUILTIN(__hipsycl_trunc, truncf, trunc)

// ***************** native math builtins ******************

#define HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(name, sp_impl, dp_impl)     \
  HIPSYCL_BUILTIN float name(float x) noexcept { return sp_impl(x); }          \
  HIPSYCL_BUILTIN double name(double x) noexcept { return dp_impl(x); }

#define HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN2(name, sp_impl, dp_impl)    \
  HIPSYCL_BUILTIN float name(float x, float y) noexcept {                      \
    return sp_impl(x, y);                                                      \
  }                                                                            \
  HIPSYCL_BUILTIN double name(double x, double y) noexcept {                   \
    return dp_impl(x, y);                                                      \
  }

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_cos, __cosf,
                                           __hipsycl_cos)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_divide(T x, T y) noexcept {
  return x / y;
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_exp, __expf,
                                           __hipsycl_exp)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp2(T x) noexcept {
  return __hipsycl_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp10(T x) noexcept {
  return __hipsycl_exp10(x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_log, __logf,
                                           __hipsycl_log)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_log2, __log2f,
                                           __hipsycl_log2)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_log10, __log10f,
                                           __hipsycl_log10)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN2(__hipsycl_native_powr, __powf,
                                           __hipsycl_powr)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_recip(T x) noexcept {
  return __hipsycl_native_divide(T{1}, x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_rsqrt, __frsqrt_rn,
                                           __hipsycl_rsqrt)
HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_sin, __sinf,
                                           __hipsycl_sin)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_sqrt(T x) noexcept {
  return __hipsycl_sqrt(x);
}

HIPSYCL_DEFINE_HIPLIKE_NATIVE_MATH_BUILTIN(__hipsycl_native_tan, __tanf,
                                           __hipsycl_tan)

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_cos(T x) noexcept {
  return __hipsycl_cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp(T x) noexcept {
  return __hipsycl_exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp2(T x) noexcept {
  return __hipsycl_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp10(T x) noexcept {
  return __hipsycl_exp10(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log(T x) noexcept {
  return __hipsycl_log(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log2(T x) noexcept {
  return __hipsycl_log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log10(T x) noexcept {
  return __hipsycl_log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_powr(T x, T y) noexcept {
  return __hipsycl_powr(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_recip(T x) noexcept {
  return __hipsycl_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_rsqrt(T x) noexcept {
  return __hipsycl_rsqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_sin(T x) noexcept {
  return __hipsycl_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_sqrt(T x) noexcept {
  return __hipsycl_sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_tan(T x) noexcept {
  return __hipsycl_tan(x);
}

// ***************** integer functions **************

template<class T, std::enable_if_t<std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_abs(T x) noexcept {
  return (x < 0) ? -x : x;
}

template<class T, std::enable_if_t<!std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_abs(T x) noexcept {
  return x;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_max(T x, T y) noexcept {
  return (x > y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_min(T x, T y) noexcept {
  return (x < y) ? x : y;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  return __hipsycl_min(__hipsycl_max(x, minval), maxval);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mul24(T x, T y) noexcept {
  return __mul24(x, y);
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  return __hipsycl_fmin(__hipsycl_fmax(x, minval), maxval);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_degrees(T x) noexcept {
  return (180.f / M_PI) * x;
}

// __hipsycl_max() and __hipsycl_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mix(T x, T y, T a) noexcept {
  return x + (y - x) * a;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_radians(T x) noexcept {
  return (M_PI / 180.f) * x;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_step(T edge, T x) noexcept {
  return (x < edge) ? T{0} : T{1};
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_smoothstep(T edge0, T edge1, T x) noexcept {
  T t = __hipsycl_clamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
  return t * t * (3 - 2 * t);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sign(T x) noexcept {
  if(std::isnan(x))
    return T{0};
  return (x == T{0}) ? x : ((x > 0) ? T{1} : T{-1});
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__hipsycl_cross3(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(),
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x()};
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__hipsycl_cross4(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(), 
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x(),
          typename VecType::element_type{0}};
}

// ****************** geometric functions ******************

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __hipsycl_dot(T a, T b) noexcept {
  return a * b;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __hipsycl_dot(T a, T b) noexcept {
  typename T::element_type result = 0;
  for (int i = 0; i < a.get_count(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __hipsycl_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __hipsycl_length(T a) noexcept {
  auto d = __hipsycl_dot(a, a);
  return __hipsycl_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __hipsycl_distance(T a, T b) noexcept {
  return __hipsycl_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_normalize(T a) noexcept {
  // TODO rsqrt might be more efficient
  return a / __hipsycl_length(a);
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __hipsycl_fast_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __hipsycl_fast_length(T a) noexcept {
  auto d = __hipsycl_dot(a, a);
  return __hipsycl_half_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __hipsycl_fast_distance(T a, T b) noexcept {
  return __hipsycl_fast_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fast_normalize(T a) noexcept {
  // TODO use rsqrt
  return a / __hipsycl_fast_length(a);
}

}
}
}

#endif

#endif
