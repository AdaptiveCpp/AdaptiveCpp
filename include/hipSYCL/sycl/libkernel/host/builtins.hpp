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

#ifndef HIPSYCL_LIBKERNEL_HOST_BUILTINS_HPP
#define HIPSYCL_LIBKERNEL_HOST_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"

#include <cstdlib>
#include <cmath>
#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST

namespace hipsycl {
namespace sycl {
namespace detail {

// ********************** math builtins *********************

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acos(T x) noexcept {
  return std::acos(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acosh(T x) noexcept {
  return std::acosh(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_acospi(T x) noexcept {
  return std::acos(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asin(T x) noexcept {
  return std::asin(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinh(T x) noexcept {
  return std::asinh(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinpi(T x) noexcept {
  return std::asin(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan(T x) noexcept {
  return std::atan(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2(T x, T y) noexcept {
  return std::atan2(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanh(T x) noexcept {
  return std::atanh(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanpi(T x) noexcept {
  return std::atan(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2pi(T x, T y) noexcept {
  return std::atan2(x, y) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cbrt(T x) noexcept {
  return std::cbrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_ceil(T x) noexcept {
  return std::ceil(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_copysign(T x, T y) noexcept {
  return std::copysign(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cos(T x) noexcept {
  return std::cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cosh(T x) noexcept {
  return std::cosh(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cospi(T x) noexcept {
  return std::cos(x * M_PI);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_erfc(T x) noexcept {
  return std::erfc(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_erf(T x) noexcept {
  return std::erf(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp(T x) noexcept {
  return std::exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp2(T x) noexcept {
  return std::exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_exp10(T x) noexcept {
  return std::pow(10, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_expm1(T x) noexcept {
  return std::expm1(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fabs(T x) noexcept {
  return std::fabs(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fdim(T x, T y) noexcept {
  return std::fdim(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_floor(T x) noexcept {
  return std::floor(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fma(T x, T y, T z) noexcept {
  return std::fma(x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmax(T x, T y) noexcept {
  return std::fmax(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmin(T x, T y) noexcept {
  return std::fmin(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_fmod(T x, T y) noexcept {
  return std::fmod(x, y);
}

template<class T>
T __hipsycl_fract(T x, T* ptr) noexcept;

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_frexp(T x, IntPtr y) noexcept {
  return std::frexp(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_hypot(T x, T y) noexcept {
  return std::hypot(x, y);
}

template<class T>
HIPSYCL_BUILTIN int __hipsycl_ilogb(T x) noexcept {
  return std::ilogb(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_lgamma(T x) noexcept {
  return std::lgamma(x);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_lgamma_r(T x, IntPtr y) noexcept {
  auto r = __hipsycl_lgamma(x);
  auto g = std::tgamma(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log(T x) noexcept {
  return std::log(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log2(T x) noexcept {
  return std::log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log10(T x) noexcept {
  return std::log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_log1p(T x) noexcept {
  return std::log1p(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_logb(T x) noexcept {
  return std::logb(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mad(T x, T y, T z) noexcept {
  return std::fma(x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_maxmag(T x, T y) noexcept {
  auto abs_x = std::abs(x);
  auto abs_y = std::abs(y);
  if(abs_x == abs_y) return std::fmax(x,y);
  return (abs_x > abs_y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_minmag(T x, T y) noexcept {
  auto abs_x = std::abs(x);
  auto abs_y = std::abs(y);
  if(abs_x == abs_y) return std::fmin(x,y);
  return (abs_x < abs_y) ? x : y;
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __hipsycl_modf(T x, FloatPtr y) noexcept {
  return std::modf(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_nextafter(T x, T y) noexcept {
  return std::nextafter(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_pow(T x, T y) noexcept {
  return std::pow(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_powr(T x, T y) noexcept {
  return std::pow(x, y);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_pown(T x, IntType y) noexcept {
  return std::pow(x, static_cast<T>(y));
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_remainder(T x, T y) noexcept {
  return std::remainder(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_rint(T x) noexcept {
  return std::rint(x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_rootn(T x, IntType y) noexcept {
  return std::pow(x, T{1}/T{y});
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_round(T x) noexcept {
  return std::round(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_rsqrt(T x) noexcept {
  return T{1} / std::sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sin(T x) noexcept {
  return std::sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sinpi(T x) noexcept {
  return std::sin(x * M_PI);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sqrt(T x) noexcept {
  return std::sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tan(T x) noexcept {
  return std::tan(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tanh(T x) noexcept {
  return std::tanh(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_tgamma(T x) noexcept {
  return std::tgamma(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_trunc(T x) noexcept {
  return std::trunc(x);
}

// ***************** native math builtins ******************

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_cos(T x) noexcept {
  return __hipsycl_cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp(T x) noexcept {
  return __hipsycl_exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp2(T x) noexcept {
  return __hipsycl_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_exp10(T x) noexcept {
  return __hipsycl_exp10(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log(T x) noexcept {
  return __hipsycl_log(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log2(T x) noexcept {
  return __hipsycl_log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_log10(T x) noexcept {
  return __hipsycl_log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_powr(T x, T y) noexcept {
  return __hipsycl_powr(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_recip(T x) noexcept {
  return __hipsycl_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_rsqrt(T x) noexcept {
  return __hipsycl_rsqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_sin(T x) noexcept {
  return __hipsycl_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_sqrt(T x) noexcept {
  return __hipsycl_sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_tan(T x) noexcept {
  return __hipsycl_tan(x);
}

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
  return std::abs(x);
}

template<class T, std::enable_if_t<!std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_abs(T x) noexcept {
  return x;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  return std::min(std::max(x, minval), maxval);
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_max(T x, T y) noexcept {
  return (x > y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_min(T x, T y) noexcept {
  return (x < y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_mul24(T x, T y) noexcept {
  return x * y;
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __hipsycl_clamp(T x, T minval, T maxval) noexcept {
  return std::fmin(std::fmax(x, minval), maxval);
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
