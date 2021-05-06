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

#ifndef HIPSYCL_LIBKERNEL_SPIRV_BUILTINS_HPP
#define HIPSYCL_LIBKERNEL_SPIRV_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"

#include <cstdlib>
#include <cmath>
#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV

namespace hipsycl {
namespace sycl {
namespace detail {

// ********************** math builtins *********************

#define HIPSYCL_DEFINE_SPIRV_BUILTIN(name)                                     \
  template <class T> HIPSYCL_BUILTIN T __hipsycl_##name(T x) noexcept {        \
    return __spirv_ocl_##name(x);                                              \
  }

#define HIPSYCL_DEFINE_SPIRV_BUILTIN2(name)                                    \
  template <class T> HIPSYCL_BUILTIN T __hipsycl_##name(T x, T y) noexcept {   \
    return __spirv_ocl_##name(x,y);                                              \
  }

#define HIPSYCL_DEFINE_SPIRV_BUILTIN3(name)                                    \
  template <class T>                                                           \
  HIPSYCL_BUILTIN T __hipsycl_##name(T x, T y, T z) noexcept {                 \
    return __spirv_ocl_##name(x,y,z);                                              \
  }

HIPSYCL_DEFINE_SPIRV_BUILTIN(acos)
HIPSYCL_DEFINE_SPIRV_BUILTIN(acosh)


template<class T>
HIPSYCL_BUILTIN T __hipsycl_acospi(T x) noexcept {
  return __hipsycl_acos(x) / M_PI;
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(asin)
HIPSYCL_DEFINE_SPIRV_BUILTIN(asinh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_asinpi(T x) noexcept {
  return __hipsycl_asin(x) / M_PI;
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(atan)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(atan2)
HIPSYCL_DEFINE_SPIRV_BUILTIN(atanh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atanpi(T x) noexcept {
  return __hipsycl_atan(x) / M_PI;
}

template<class T>
HIPSYCL_BUILTIN T __hipsycl_atan2pi(T x, T y) noexcept {
  return __hipsycl_atan2(x, y) / M_PI;
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(cbrt)
HIPSYCL_DEFINE_SPIRV_BUILTIN(ceil)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(copysign)
HIPSYCL_DEFINE_SPIRV_BUILTIN(cos)
HIPSYCL_DEFINE_SPIRV_BUILTIN(cosh)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_cospi(T x) noexcept {
  return __hipsycl_cos(x * M_PI);
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(erf)
HIPSYCL_DEFINE_SPIRV_BUILTIN(erfc)
HIPSYCL_DEFINE_SPIRV_BUILTIN(exp)
HIPSYCL_DEFINE_SPIRV_BUILTIN(exp2)
HIPSYCL_DEFINE_SPIRV_BUILTIN(exp10)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(pow)


HIPSYCL_DEFINE_SPIRV_BUILTIN(expm1)
HIPSYCL_DEFINE_SPIRV_BUILTIN(fabs)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(fdim)
HIPSYCL_DEFINE_SPIRV_BUILTIN(floor)
HIPSYCL_DEFINE_SPIRV_BUILTIN3(fma)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(fmax)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(fmin)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(fmod)

// Unsupported
template<class T>
T __hipsycl_fract(T x, T* ptr) noexcept;

// Unsupported
template<class T, class IntPtr>
T __hipsycl_frexp(T x, IntPtr y) noexcept;

HIPSYCL_DEFINE_SPIRV_BUILTIN2(hypot)
HIPSYCL_DEFINE_SPIRV_BUILTIN(ilogb)
HIPSYCL_DEFINE_SPIRV_BUILTIN(lgamma)
HIPSYCL_DEFINE_SPIRV_BUILTIN(tgamma)

template<class T, class IntPtr>
HIPSYCL_BUILTIN T __hipsycl_lgamma_r(T x, IntPtr y) noexcept {
  return __spirv_ocl_lgamma_r(x, y);
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(log)
HIPSYCL_DEFINE_SPIRV_BUILTIN(log2)
HIPSYCL_DEFINE_SPIRV_BUILTIN(log10)
HIPSYCL_DEFINE_SPIRV_BUILTIN(log1p)
HIPSYCL_DEFINE_SPIRV_BUILTIN(logb)

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

HIPSYCL_DEFINE_SPIRV_BUILTIN2(nextafter)
HIPSYCL_DEFINE_SPIRV_BUILTIN2(powr)


template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_pown(T x, IntType y) noexcept {
  return __spirv_ocl_pown(x, y);
}

HIPSYCL_DEFINE_SPIRV_BUILTIN2(remainder)
HIPSYCL_DEFINE_SPIRV_BUILTIN(rint)

template<class T, class IntType>
HIPSYCL_BUILTIN T __hipsycl_rootn(T x, IntType y) noexcept {
  return __spirv_ocl_rootn(x, y);
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(round)
HIPSYCL_DEFINE_SPIRV_BUILTIN(rsqrt)
HIPSYCL_DEFINE_SPIRV_BUILTIN(sqrt)
HIPSYCL_DEFINE_SPIRV_BUILTIN(sin)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_sinpi(T x) noexcept {
  return __hipsycl_sin(x * M_PI);
}

HIPSYCL_DEFINE_SPIRV_BUILTIN(tan)
HIPSYCL_DEFINE_SPIRV_BUILTIN(tanh)
HIPSYCL_DEFINE_SPIRV_BUILTIN(trunc)

// ***************** native math builtins ******************

#define HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(name)                     \
  template <class T> HIPSYCL_BUILTIN T __hipsycl_native_##name(T x) noexcept { \
    return __hipsycl_##name(x);                                                \
  }

#define HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN2(name)                    \
  template <class T>                                                           \
  HIPSYCL_BUILTIN T __hipsycl_native_##name(T x, T y) noexcept {               \
    return __hipsycl_##name(x, y);                                             \
  }

HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(cos)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_divide(T x, T y) noexcept {
  return x / y;
}

HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(exp)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(exp2)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(exp10)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(log)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(log2)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(log10)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN2(powr)

template<class T>
HIPSYCL_BUILTIN T __hipsycl_native_recip(T x) noexcept {
  return __hipsycl_native_divide(T{1}, x);
}

HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(rsqrt)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(sin)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(sqrt)
HIPSYCL_DEFINE_SPIRV_FALLBACK_NATIVE_BUILTIN(tan)

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

template<class T, std::enable_if_t<std::is_signed_v<T>, int> = 0>
HIPSYCL_BUILTIN T __hipsycl_mul24(T x, T y) noexcept {
  return __spirv_ocl_s_mul24(x, y);
}

template<class T, std::enable_if_t<!std::is_signed_v<T>, int> = 0>
HIPSYCL_BUILTIN T __hipsycl_mul24(T x, T y) noexcept {
  return __spirv_ocl_u_mul24(x, y);
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
  if(__spirv_IsNan(x))
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
