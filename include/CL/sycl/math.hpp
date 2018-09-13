/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_MATH_HPP
#define HIPSYCL_MATH_HPP

#include "backend/backend.hpp"
#include <type_traits>
#include <cmath>
#include "vec.hpp"

namespace cl {
namespace sycl {

#define HIPSYCL_ENABLE_IF_FLOATING_POINT(template_param) \
  std::enable_if_t<std::is_floating_point<float_type>::value>* = nullptr

namespace detail {

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type acospi(float_type x)
{return std::acos(x)/static_cast<float_type>(M_PI); }


template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type asinpi(float_type x)
{return std::asin(x)/static_cast<float_type>(M_PI); }


template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type atanpi(float_type x)
{return std::atan(x)/static_cast<float_type>(M_PI); }


template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type atan2pi(float_type y, float_type x)
{return std::atan2(y,x)/static_cast<float_type>(M_PI); }

// ToDo: Use cuda's cospi() when on NVIDIA
template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type cospi(float_type x)
{ return std::cos(x * static_cast<float_type>(M_PI)); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type sinpi(float_type x)
{ return std::sin(x * static_cast<float_type>(M_PI)); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type tanpi(float_type x)
{ return std::tan(x * static_cast<float_type>(M_PI)); }


__device__
inline float exp10(float x)
{ return ::exp10f(x); }

__device__
inline double exp10(double x)
{ return ::exp10(x); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type fabs(float_type x)
{ return std::abs(x); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type fmax(float_type x, float_type y)
{ return std::max(x,y); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type fmin(float_type x, float_type y)
{ return std::min(x,y); }

__device__
inline float logb(float x)
{ return ::logbf(x); }

__device__
inline double logb(double x)
{ return ::logb(x); }

template<typename float_type,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline float_type rsqrt(float_type x)
{ return static_cast<float_type>(1.f) / std::sqrt(x); }



} // detail


#define HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(name, func) \
  template<class float_type, int N,\
           HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)> \
  __device__ \
  inline vec<float_type,N> name(const vec<float_type, N>& v) {\
    vec<float_type,N> result = v; \
    detail::transform_vector(result, \
                      (float_type (*)(float_type))&func); \
    return result; \
  }

#define HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(name, func) \
  template<class float_type, int N, \
           HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)> \
  __device__ \
  inline vec<float_type, N> name(const vec<float_type, N>& a, \
                                 const vec<float_type, N>& b) {\
    return detail::binary_vector_operation(a,b,\
                          (float_type (*)(float_type,float_type))&func); \
  }

#define HIPSYCL_DEFINE_FLOATN_TRINARY_MATH_FUNCTION(name, func) \
  template<class float_type, int N, \
           HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)> \
  __device__ \
  inline vec<float_type, N> name(const vec<float_type, N>& a, \
                                 const vec<float_type, N>& b, \
                                 const vec<float_type, N>& c) {\
    return detail::trinary_vector_operation(a,b,c,\
               (float_type (*)(float_type,float_type,float_type))&func); \
  }

#define HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(func) \
  using ::std::func; \
  HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(func, ::std::func)

#define HIPSYCL_DEFINE_GENFLOAT_BINARY_STD_FUNCTION(func) \
  using ::std::func; \
  HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(func, ::std::func)


HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(acos)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(acosh)

using detail::acospi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(acospi, detail::acospi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(asin)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(asinh)

using detail::asinpi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(asinpi, detail::asinpi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(atan)
HIPSYCL_DEFINE_GENFLOAT_BINARY_STD_FUNCTION(atan2)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(atanh)

using detail::atanpi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(atanpi, detail::atanpi)

using detail::atan2pi;
HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(atan2pi, detail::atan2pi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(cbrt)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(ceil)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(copysign)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(cos)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(cosh)

using detail::cospi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(cospi, detail::cospi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(erf)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(erfc)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(exp)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(exp2)

using detail::exp10;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(exp10, detail::exp10)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(expm1)

using detail::fabs;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(fabs, detail::fabs)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(fdim)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(floor)

// ToDo: Triple Op
using std::fma;
HIPSYCL_DEFINE_FLOATN_TRINARY_MATH_FUNCTION(fma, std::fma);

using detail::fmin;
using detail::fmax;
HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(fmin, detail::fmin)
HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(fmax, detail::fmax)

template<class float_type, int N,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline vec<float_type, N> fmin(const vec<float_type, N>& a,
                               float_type b) {
  return fmin(a, vec<float_type,N>{b});
}

template<class float_type, int N,
         HIPSYCL_ENABLE_IF_FLOATING_POINT(float_type)>
__device__
inline vec<float_type, N> fmax(const vec<float_type, N>& a,
                               float_type b) {
  return fmax(a, vec<float_type,N>{b});
}

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(fmod)

// ToDo fract
// ToDo frexp

HIPSYCL_DEFINE_GENFLOAT_BINARY_STD_FUNCTION(hypot)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(ilogb)

// ToDo ldexp

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(lgamma)

// ToDo lgamma_r

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(log)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(log2)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(log10)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(log1p)

using detail::logb;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(logb, detail::logb)

// ToDo mad - unsupported in cuda/hip

// ToDo maxmag
// ToDo minmag
// ToDo modf

// ToDo nan

// ToDo nextafter

HIPSYCL_DEFINE_GENFLOAT_BINARY_STD_FUNCTION(pow)
// ToDo pown
// ToDo powr

// ToDo remainder
// ToDo remquo
// ToDo rint
// ToDo rootn

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(round)

using detail::rsqrt;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(rsqrt, detail::rsqrt)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(sin)

// ToDo sincos

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(sinh)

using detail::sinpi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(sinpi, detail::sinpi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(sqrt)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(tan)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(tanh)

using detail::tanpi;
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(tanpi, detail::tanpi)

HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(tgamma)
HIPSYCL_DEFINE_GENFLOAT_STD_FUNCTION(trunc)

namespace native {


#define HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(name, fallback_func, fast_sp_func) \
  template<class float_type> \
  __device__ inline float_type name(float_type x){return fallback_func(x);} \
  template<> \
  __device__ inline float name(float x) { return fast_sp_func(x); }

#define HIPSYCL_DEFINE_FAST_FUNCTION(name, fallback_func, \
                                     fast_sp_func,\
                                     fast_dp_func) \
  template<class float_type> \
  __device__ inline float_type name(float_type x){return fallback_func(x);} \
  template<> \
  __device__ inline float name(float x) { return fast_sp_func(x); } \
  template<> \
  __device__ inline double name(double x) { return fast_dp_func(x); } \

HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(cos, std::cos, __cosf);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(exp, std::exp, __expf);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(rsqrt, std::exp, __frsqrt_rn);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(log10, std::log10, __log10f);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(log2, std::log2, __log2f);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(log, std::log, __logf);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(sin, std::sin, __sinf);
HIPSYCL_DEFINE_FAST_SINGLE_PRECISION_FUNCTION(tan, std::tan, __tanf);
HIPSYCL_DEFINE_FAST_FUNCTION(sqrt, std::sqrt, __fsqrt_rn, __dsqrt_rn);

template<class float_type>
__device__
inline float_type pow(float_type x, float_type y){return std::pow(x,y);}

template<>
__device__
inline float pow(float x, float y) { return __powf(x,y); }

HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(cos, cos);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(exp, exp);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(rsqrt, rsqrt);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(log10, log10);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(log2, log2);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(log, log);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(sin, sin);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(tan, tan);
HIPSYCL_DEFINE_FLOATN_MATH_FUNCTION(sqrt, sqrt);
HIPSYCL_DEFINE_FLOATN_BINARY_MATH_FUNCTION(pow, pow);

} // native

namespace half_precision {

// ToDo

} // half_precision

} // sycl
} // cl

#endif
