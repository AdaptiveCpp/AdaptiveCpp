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
#ifndef HIPSYCL_BUILTIN_INTERFACE_HPP
#define HIPSYCL_BUILTIN_INTERFACE_HPP

#include "detail/builtin_dispatch.hpp"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/builtins.hpp"
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP ||                                    \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
#include "generic/hiplike/builtins.hpp"
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/builtins.hpp"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {


template<class T>
HIPSYCL_BUILTIN T __acpp_acos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_acos, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_acosh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_acosh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_acospi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_acospi, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_asin, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asinh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_asinh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asinpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_asinpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atan, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan2(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atan2, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atanh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atanh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atanpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atanpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan2pi(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atan2pi, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cbrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cbrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_ceil(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_ceil, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_copysign(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_copysign, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cosh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cosh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cospi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cospi, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_erfc(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_erfc, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_erf(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_erf, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_expm1(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_expm1, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fabs(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fabs, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fdim(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fdim, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_floor(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_floor, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fma(T x, T y, T z) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fma, x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmax(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fmax, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmin(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fmin, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmod(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fmod, x, y);
}

template<class T>
T __acpp_fract(T x, T* ptr) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fract, x, ptr);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __acpp_frexp(T x, IntPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_frexp, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_hypot(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_hypot, x, y);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_ilogb(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_ilogb, x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_ldexp(T x, IntType k) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_ldexp, x, k);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_lgamma(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_acos, x);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __acpp_lgamma_r(T x, IntPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_lgamma_r, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log1p(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_log1p, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_logb(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_logb, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_mad(T x, T y, T z) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_mad, x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_maxmag(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_maxmag, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_minmag(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_minmag, x, y);
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __acpp_modf(T x, FloatPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_modf, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_nextafter(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_nextafter, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_pow(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_pow, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_powr, x, y);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_pown(T x, IntType y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_pown, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_remainder(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_remainder, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_rint(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_rint, x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_rootn(T x, IntType y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_rootn, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_round(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_round, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sin, x);
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __acpp_sincos(T x, FloatPtr cosval) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sincos, x, cosval);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sinh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sinh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sinpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sinpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_tan, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tanh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_tanh, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tgamma(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_tgamma, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_trunc(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_trunc, x);
}

// ***************** native math builtins ******************

template<class T>
HIPSYCL_BUILTIN T __acpp_native_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_divide(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_divide, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_powr, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_recip(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_recip, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_sin, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_native_tan, x);
}

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_BUILTIN T __acpp_half_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_divide(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_divide, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_powr, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_recip(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_recip, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_sin, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_half_tan, x);
}

// ***************** integer functions **************

template<class T>
HIPSYCL_BUILTIN T __acpp_abs(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_abs, x);
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_clamp, x, minval, maxval);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_ctz, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_clz, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_max(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_max, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_min(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_min, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_mul24(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_mul24, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_popcount(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_popcount, x);
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_clamp, x, minval, maxval);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_degrees(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_degrees, x);
}

// __acpp_max() and __acpp_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_BUILTIN T __acpp_mix(T x, T y, T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_mix, x, y, a);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_radians(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_radians, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_step(T edge, T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_step, edge, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_smoothstep(T edge0, T edge1, T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_smoothstep, edge0, edge1, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sign(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_sign, x);
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross3(const VecType &a, const VecType &b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cross3, a, b);
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross4(const VecType &a, const VecType &b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_cross4, a, b);
}

// ****************** geometric functions ******************

template <class T>
HIPSYCL_BUILTIN auto __acpp_dot(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_dot, a, b);
}

template <class T>
HIPSYCL_BUILTIN auto __acpp_length(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_length, a);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_distance(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_distance, a, b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_normalize(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_normalize, a);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_fast_length(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fast_length, a);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_fast_distance(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fast_distance, a, b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fast_normalize(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_fast_normalize, a);
}

// ****************** relational functions ******************

template<class T>
HIPSYCL_BUILTIN int __acpp_isnan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_isnan, x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isinf(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_isinf, x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isfinite(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_isfinite, x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isnormal(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_isnormal, x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_signbit(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_signbit, x);
}

}
}
}

#endif
