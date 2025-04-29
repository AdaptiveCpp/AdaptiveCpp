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

#ifndef ACPP_LIBKERNEL_HOST_BUILTINS_HPP
#define ACPP_LIBKERNEL_HOST_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#ifndef HIPSYCL_SSCP_LIBKERNEL_LIBRARY
#include "hipSYCL/sycl/libkernel/vec.hpp"
#endif

#include <bitset>
#include <cstdlib>
#include <cmath>
#include <type_traits>
#include <climits>

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST

namespace hipsycl {
namespace sycl {
namespace detail::host_builtins {

// ********************** math builtins *********************

template<class T>
HIPSYCL_BUILTIN T __acpp_acos(T x) noexcept {
  return std::acos(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_acosh(T x) noexcept {
  return std::acosh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_acospi(T x) noexcept {
  return std::acos(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asin(T x) noexcept {
  return std::asin(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asinh(T x) noexcept {
  return std::asinh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_asinpi(T x) noexcept {
  return std::asin(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan(T x) noexcept {
  return std::atan(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan2(T x, T y) noexcept {
  return std::atan2(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atanh(T x) noexcept {
  return std::atanh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atanpi(T x) noexcept {
  return std::atan(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_atan2pi(T x, T y) noexcept {
  return std::atan2(x, y) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cbrt(T x) noexcept {
  return std::cbrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_ceil(T x) noexcept {
  return std::ceil(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_copysign(T x, T y) noexcept {
  return std::copysign(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cos(T x) noexcept {
  return std::cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cosh(T x) noexcept {
  return std::cosh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_cospi(T x) noexcept {
  return std::cos(x * M_PI);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_erfc(T x) noexcept {
  return std::erfc(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_erf(T x) noexcept {
  return std::erf(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp(T x) noexcept {
  return std::exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp2(T x) noexcept {
  return std::exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_exp10(T x) noexcept {
  return std::pow(10, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_expm1(T x) noexcept {
  return std::expm1(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fabs(T x) noexcept {
  return std::fabs(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fdim(T x, T y) noexcept {
  return std::fdim(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_floor(T x) noexcept {
  return std::floor(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fma(T x, T y, T z) noexcept {
  return std::fma(x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmax(T x, T y) noexcept {
  return std::fmax(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmin(T x, T y) noexcept {
  return std::fmin(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fmod(T x, T y) noexcept {
  return std::fmod(x, y);
}

template<class T>
T __acpp_fract(T x, T* ptr) noexcept;

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __acpp_frexp(T x, IntPtr y) noexcept {
  return std::frexp(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_hypot(T x, T y) noexcept {
  return std::hypot(x, y);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_ilogb(T x) noexcept {
  return std::ilogb(x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_ldexp(T x, IntType k) noexcept {
  return std::ldexp(x, k);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_lgamma(T x) noexcept {
  return std::lgamma(x);
}

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __acpp_lgamma_r(T x, IntPtr y) noexcept {
  auto r = host_builtins::__acpp_lgamma(x);
  auto g = std::tgamma(x);
  *y = (g >= 0) ? 1 : -1;
  return r;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log(T x) noexcept {
  return std::log(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log2(T x) noexcept {
  return std::log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log10(T x) noexcept {
  return std::log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_log1p(T x) noexcept {
  return std::log1p(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_logb(T x) noexcept {
  return std::logb(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_mad(T x, T y, T z) noexcept {
  return std::fma(x, y, z);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_maxmag(T x, T y) noexcept {
  auto abs_x = std::abs(x);
  auto abs_y = std::abs(y);
  if(abs_x == abs_y) return std::fmax(x,y);
  return (abs_x > abs_y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_minmag(T x, T y) noexcept {
  auto abs_x = std::abs(x);
  auto abs_y = std::abs(y);
  if(abs_x == abs_y) return std::fmin(x,y);
  return (abs_x < abs_y) ? x : y;
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __acpp_modf(T x, FloatPtr y) noexcept {
  return std::modf(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_nextafter(T x, T y) noexcept {
  return std::nextafter(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_pow(T x, T y) noexcept {
  return std::pow(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_powr(T x, T y) noexcept {
  return std::pow(x, y);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_pown(T x, IntType y) noexcept {
  return std::pow(x, static_cast<T>(y));
}

template<class T>
HIPSYCL_BUILTIN T __acpp_remainder(T x, T y) noexcept {
  return std::remainder(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_rint(T x) noexcept {
  return std::rint(x);
}

template<class T, class IntType>
HIPSYCL_BUILTIN T __acpp_rootn(T x, IntType y) noexcept {
  return std::pow(x, T{1}/T{y});
}

template<class T>
HIPSYCL_BUILTIN T __acpp_round(T x) noexcept {
  return std::round(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_rsqrt(T x) noexcept {
  return T{1} / std::sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sin(T x) noexcept {
  return std::sin(x);
}

template<class T, class FloatPtr>
HIPSYCL_BUILTIN T __acpp_sincos(T x, FloatPtr cosval) noexcept {
  *cosval = host_builtins::__acpp_cos(x);
  return  host_builtins::__acpp_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sinh(T x) noexcept {
  return std::sinh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sinpi(T x) noexcept {
  return std::sin(x * M_PI);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sqrt(T x) noexcept {
  return std::sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tan(T x) noexcept {
  return std::tan(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tanh(T x) noexcept {
  return std::tanh(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_tgamma(T x) noexcept {
  return std::tgamma(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_trunc(T x) noexcept {
  return std::trunc(x);
}

// ***************** native math builtins ******************

template<class T>
HIPSYCL_BUILTIN T __acpp_native_cos(T x) noexcept {
  return __acpp_cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp(T x) noexcept {
  return host_builtins::__acpp_exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp2(T x) noexcept {
  return host_builtins::__acpp_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_exp10(T x) noexcept {
  return host_builtins::__acpp_exp10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log(T x) noexcept {
  return host_builtins::__acpp_log(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log2(T x) noexcept {
  return host_builtins::__acpp_log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_log10(T x) noexcept {
  return host_builtins::__acpp_log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_powr(T x, T y) noexcept {
  return host_builtins::__acpp_powr(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_recip(T x) noexcept {
  return host_builtins::__acpp_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_rsqrt(T x) noexcept {
  return host_builtins::__acpp_rsqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_sin(T x) noexcept {
  return host_builtins::__acpp_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_sqrt(T x) noexcept {
  return host_builtins::__acpp_sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_native_tan(T x) noexcept {
  return host_builtins::__acpp_tan(x);
}

// ***************** half precision math builtins ******************


template<class T>
HIPSYCL_BUILTIN T __acpp_half_cos(T x) noexcept {
  return host_builtins::__acpp_cos(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_divide(T x, T y) noexcept {
  return x / y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp(T x) noexcept {
  return host_builtins::__acpp_exp(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp2(T x) noexcept {
  return host_builtins::__acpp_exp2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_exp10(T x) noexcept {
  return host_builtins::__acpp_exp10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log(T x) noexcept {
  return host_builtins::__acpp_log(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log2(T x) noexcept {
  return host_builtins::__acpp_log2(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_log10(T x) noexcept {
  return host_builtins::__acpp_log10(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_powr(T x, T y) noexcept {
  return host_builtins::__acpp_powr(x, y);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_recip(T x) noexcept {
  return host_builtins::__acpp_native_divide(T{1}, x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_rsqrt(T x) noexcept {
  return host_builtins::__acpp_rsqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sin(T x) noexcept {
  return host_builtins::__acpp_sin(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_sqrt(T x) noexcept {
  return host_builtins::__acpp_sqrt(x);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_half_tan(T x) noexcept {
  return host_builtins::__acpp_tan(x);
}

// ***************** integer functions **************

template<class T, std::enable_if_t<std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_abs(T x) noexcept {
  return std::abs(x);
}

template<class T, std::enable_if_t<!std::is_signed_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_abs(T x) noexcept {
  return x;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return std::min(std::max(x, minval), maxval);
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
inline T fallback_ctz(T x) noexcept {

  if(x==0){return sizeof(T)*CHAR_BIT;}
  std::bitset<sizeof(T)*CHAR_BIT> bset(x);
  int idx = 0;
  while(!bset[idx]){idx++;}
  return idx;

}

template <class T,
          std::enable_if_t<
              (std::is_same_v<T, unsigned int> || std::is_same_v<T, int> ||
               std::is_same_v<T, unsigned short> || std::is_same_v<T, short> ||
               std::is_same_v<T, unsigned char> ||
               std::is_same_v<T, signed char> || std::is_same_v<T, char>),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {

  #if __has_builtin(__builtin_ctz)
    // builtin_ctz(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    //we convert to the unsigned type to avoid the typecast creating 
    //additional ones in front of the value if x is negative
    using Usigned = typename std::make_unsigned<T>::type; 
    return __builtin_ctz(static_cast<Usigned>(x));
  #else
    return fallback_ctz(x);
  #endif
}

template <class T, std::enable_if_t<(std::is_same_v<T, unsigned long> ||
                                     std::is_same_v<T, long>),
                                    int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  #if __has_builtin(__builtin_ctzl)
    // builtin_ctzl(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    return __builtin_ctzl(static_cast<unsigned long>(x));
  #else
    return fallback_ctz(x);
  #endif
}

template <class T, std::enable_if_t<(std::is_same_v<T, unsigned long long> ||
                                     std::is_same_v<T, long long>),
                                    int> = 0>
HIPSYCL_BUILTIN T __acpp_ctz(T x) noexcept {
  #if __has_builtin(__builtin_ctzll)
    // builtin_ctzll(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    return __builtin_ctzll(static_cast<unsigned long long>(x));
  #else
    return fallback_ctz(x);
  #endif
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
inline T fallback_clz(T x) noexcept {

  if(x==0){return sizeof(T)*CHAR_BIT;}
  std::bitset<sizeof(T)*CHAR_BIT> bset(x);
  int idx = 0;
  while(!bset[sizeof(T)*CHAR_BIT - idx -1]){idx++;}
  return idx;

}

template <class T,
          std::enable_if_t<
              (std::is_same_v<T, unsigned int> || std::is_same_v<T, int> ||
               std::is_same_v<T, unsigned short> || std::is_same_v<T, short> ||
               std::is_same_v<T, unsigned char> ||
               std::is_same_v<T, signed char> || std::is_same_v<T, char>),
              int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {

  #if __has_builtin(__builtin_clz)
    // builtin_clz(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    //we convert to the unsigned type to avoid the typecast creating 
    //additional ones in front of the value if x is negative
    using Usigned = typename std::make_unsigned<T>::type; 
    constexpr T diff = CHAR_BIT*(sizeof(unsigned int) - sizeof(Usigned));
    return __builtin_clz(static_cast<Usigned>(x)) - diff;
  #else
    return fallback_clz(x);
  #endif
}

template <class T, std::enable_if_t<(std::is_same_v<T, unsigned long> ||
                                     std::is_same_v<T, long>),
                                    int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  #if __has_builtin(__builtin_clzl)
    // builtin_clzl(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    return __builtin_clzl(static_cast<unsigned long>(x));
  #else
    return fallback_clz(x);
  #endif
}

template <class T, std::enable_if_t<(std::is_same_v<T, unsigned long long> ||
                                     std::is_same_v<T, long long>),
                                    int> = 0>
HIPSYCL_BUILTIN T __acpp_clz(T x) noexcept {
  #if __has_builtin(__builtin_clzll)
    // builtin_clzll(0) is UB on some arch
    if(x==0){return sizeof(T)*CHAR_BIT;}

    return __builtin_clzll(static_cast<unsigned long long>(x));
  #else
    return fallback_clz(x);
  #endif
}

template<class T>
HIPSYCL_BUILTIN T __acpp_max(T x, T y) noexcept {
  return (x > y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_min(T x, T y) noexcept {
  return (x < y) ? x : y;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_mul24(T x, T y) noexcept {
  return x * y;
}

template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_popcount(T x) noexcept {
    T count = T(0);
    for (size_t i = 0; i < sizeof(T) * 8; i++) {
        count += !!(x & (T(1) << i));
    }
    return count;
}

// **************** common functions *****************

template<class T, std::enable_if_t<!std::is_integral_v<T>,int> = 0>
HIPSYCL_BUILTIN T __acpp_clamp(T x, T minval, T maxval) noexcept {
  return std::fmin(std::fmax(x, minval), maxval);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_degrees(T x) noexcept {
  return (180.f / M_PI) * x;
}

// __acpp_max() and __acpp_min() are handled by the overloads from the
// integer functions

template<class T>
HIPSYCL_BUILTIN T __acpp_mix(T x, T y, T a) noexcept {
  return x + (y - x) * a;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_radians(T x) noexcept {
  return (M_PI / 180.f) * x;
}

template<class T>
HIPSYCL_BUILTIN T __acpp_step(T edge, T x) noexcept {
  return (x < edge) ? T{0} : T{1};
}

template<class T>
HIPSYCL_BUILTIN T __acpp_smoothstep(T edge0, T edge1, T x) noexcept {
  T t =
      host_builtins::__acpp_clamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
  return t * t * (3 - 2 * t);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_sign(T x) noexcept {
  if(std::isnan(x))
    return T{0};
  return (x == T{0}) ? x : ((x > 0) ? T{1} : T{-1});
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross3(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(),
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x()};
}

template <typename VecType>
HIPSYCL_BUILTIN VecType 
__acpp_cross4(const VecType &a, const VecType &b) noexcept {
  return {a.y() * b.z() - a.z() * b.y(), 
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x(),
          typename VecType::element_type{0}};
}

// ****************** geometric functions ******************

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_dot(T a, T b) noexcept {
  return a * b;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_dot(T a, T b) noexcept {
  typename T::element_type result = 0;
  for (int i = 0; i < a.size(); ++i) {
    result += a[i] * b[i];
  }
  return result;
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_length(T a) noexcept {
  auto d = host_builtins::__acpp_dot(a, a);
  return host_builtins::__acpp_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_distance(T a, T b) noexcept {
  return host_builtins::__acpp_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_normalize(T a) noexcept {
  // TODO rsqrt might be more efficient
  return a / host_builtins::__acpp_length(a);
}

template <class T, std::enable_if_t<std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN T __acpp_fast_length(T a) noexcept {
  return (a < 0) ? -a : a;
}

template <class T, std::enable_if_t<!std::is_arithmetic_v<T>, int> = 0>
HIPSYCL_BUILTIN typename T::element_type __acpp_fast_length(T a) noexcept {
  auto d = host_builtins::__acpp_dot(a, a);
  return host_builtins::__acpp_half_sqrt(d);
}

template<class T>
HIPSYCL_BUILTIN auto __acpp_fast_distance(T a, T b) noexcept {
  return host_builtins::__acpp_fast_length(a - b);
}

template<class T>
HIPSYCL_BUILTIN T __acpp_fast_normalize(T a) noexcept {
  // TODO use rsqrt
  return a / host_builtins::__acpp_fast_length(a);
}

// ****************** relational functions ******************

template<class T>
HIPSYCL_BUILTIN int __acpp_isnan(T x) noexcept {
  return std::isnan(x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isinf(T x) noexcept {
  return std::isinf(x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isfinite(T x) noexcept {
  return std::isfinite(x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_isnormal(T x) noexcept {
  return std::isnormal(x);
}

template<class T>
HIPSYCL_BUILTIN int __acpp_signbit(T x) noexcept {
  return std::signbit(x);
}

}
}
}

#endif

#endif
