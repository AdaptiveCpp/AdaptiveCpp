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

#ifndef HIPSYCL_BUILTIN_INTERFACE_HPP
#define HIPSYCL_BUILTIN_INTERFACE_HPP

#include "detail/builtin_dispatch.hpp"

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/builtins.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
#include "spirv/builtins.hpp"
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP ||                                    \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
#include "generic/hiplike/builtins.hpp"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {


template<class T>
HIPSYCL_BUILTIN T __hipsycl_acos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_acos, x);
}

template<class T>
HIPSYCL_BUILTIN int __hipsycl_isnan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_isnan, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acosh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_acosh, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acospi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_acospi, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_asin, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_asinh, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_asinpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atan, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atan2, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atanh, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atanpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2pi(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atan2pi, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cbrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cbrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_ceil(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_ceil, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_copysign(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_copysign, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cosh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cosh, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cospi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cospi, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_erfc(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_erfc, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_erf(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_erf, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_expm1(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_expm1, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fabs(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fabs, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fdim(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fdim, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_floor(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_floor, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fma(T x, T y, T z) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fma, x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmax(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fmax, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmin(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fmin, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmod(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fmod, x, y);
}

template<class T>
T __hipsycl_fract(T x, T* ptr) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fract, x, ptr);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_frexp(T x, IntPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_frexp, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_hypot(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_hypot, x, y);
}

template<class T>
HIPSYCL_BUILTIN int __hipsycl_ilogb(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_ilogb, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_lgamma(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_acos, x);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_lgamma_r(T x, IntPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_lgamma_r, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log1p(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_log1p, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_logb(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_logb, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mad(T x, T y, T z) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_mad, x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_maxmag(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_maxmag, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_minmag(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_minmag, x, y);
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __hipsycl_modf(T x, FloatPtr y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_modf, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_nextafter(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_nextafter, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_pow(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_pow, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_powr, x, y);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_pown(T x, IntType y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_pown, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_remainder(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_remainder, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_rint(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_rint, x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_rootn(T x, IntType y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_rootn, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_round(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_round, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_sin, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sinpi(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_sinpi, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_tan, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tanh(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_tanh, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tgamma(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_tgamma, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_trunc(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_trunc, x);
}

// ***************** native math builtins ******************

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_divide(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_divide, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_powr, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_recip(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_recip, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_sin, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_native_tan, x);
}

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_cos(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_cos, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_divide(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_divide, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_exp, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_exp2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_exp10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_exp10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_log, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log2(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_log2, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_log10(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_log10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_powr(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_powr, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_recip(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_recip, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_rsqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_rsqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_sin(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_sin, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_sqrt(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_sqrt, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_half_tan(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_half_tan, x);
}

// ***************** integer functions **************

template<class T>
HIPSYCL_BUILTIN T __hipsycl_abs(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_abs, x);
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_clamp, x, minval, maxval);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_max(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_max, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_min(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_min, x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mul24(T x, T y) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_mul24, x, y);
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_clamp, x, minval, maxval);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_degrees(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_degrees, x);
}

// __hipsycl_max() and __hipsycl_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mix(T x, T y, T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_mix, x, y, a);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_radians(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_radians, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_step(T edge, T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_step, edge, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_smoothstep(T edge0, T edge1, T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_smoothstep, edge0, edge1, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sign(T x) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_sign, x);
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__hipsycl_cross3(const VecType &a, const VecType &b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cross3, a, b);
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__hipsycl_cross4(const VecType &a, const VecType &b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_cross4, a, b);
}

// ****************** geometric functions ******************

template <class T>
HIPSYCL_BUILTIN auto __hipsycl_dot(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_dot, a, b);
}

template <class T>
HIPSYCL_BUILTIN auto __hipsycl_length(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_length, a);
}

template<class T>
HIPSYCL_BUILTIN auto __hipsycl_distance(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_distance, a, b);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_normalize(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_normalize, a);
}

template<class T>
HIPSYCL_BUILTIN auto __hipsycl_fast_length(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fast_length, a);
}

template<class T>
HIPSYCL_BUILTIN auto __hipsycl_fast_distance(T a, T b) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fast_distance, a, b);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fast_normalize(T a) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_fast_normalize, a);
}

}
}
}

#endif
