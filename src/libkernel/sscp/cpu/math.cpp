/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"
#include "hipSYCL/sycl/libkernel/host/builtins.hpp"

using namespace hipsycl::sycl::detail::host_builtins;

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(name)                   \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x) {            \
    return __hipsycl_##name(x);                                                 \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x) {          \
    return __hipsycl_##name(x);                                                 \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(name)                  \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x, float y) {   \
    return __hipsycl_##name(x, y);                                              \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x,            \
                                                          double y) {          \
    return __hipsycl_##name(x, y);                                              \
  }

#define HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3(name)                  \
                                                                               \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x, float y,     \
                                                         float z) {            \
    return __hipsycl_##name(x, y, z);                                           \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x, double y,  \
                                                          double z) {          \
    return __hipsycl_##name(x, y, z);                                           \
  }

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(acos)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(acosh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(acospi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(asin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(asinh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(asinpi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(atan)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(atan2)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(atanh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(atanpi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(atan2pi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cbrt)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(ceil)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(copysign)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cos)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cosh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(cospi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(erf)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(erfc)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp2)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(exp10)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(pow)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(expm1)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(fabs)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fdim)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(floor)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3(fma)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmax)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(fmod)

// fmin(x - floor(x), nextafter(genfloat(1.0), genfloat(0.0)) ). floor(x) is returned in iptr.
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_fract_f32(float x, float* y) {
  return __hipsycl_fract(x, y);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_fract_f64(double x, double* y) {
  return __hipsycl_fract(x, y);
}

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_frexp_f32(float x,
                                                    __hipsycl_int32 *y) {
  return __hipsycl_frexp(x, y);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_frexp_f64(double x,
                                                     __hipsycl_int64 *y) {
  __hipsycl_int32 w;
  auto res = __hipsycl_frexp(x, &w);
  *y = w;
  return res;
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(hypot)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(ilogb)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_ldexp_f32(float x,
                                                    __hipsycl_int32 k) {
  return __hipsycl_ldexp(x, k);
}

HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_ldexp_f64(double x,
                                                     __hipsycl_int64 k) {
  return __hipsycl_ldexp(x, k);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tgamma)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(lgamma)


HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_lgamma_r_f32(float x, __hipsycl_int32* y ) {
  return __hipsycl_lgamma_r(x,y);
}

HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_lgamma_r_f64(double x, __hipsycl_int64* y) {
  return __hipsycl_lgamma_r(x,y);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log2)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log10)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(log1p)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(logb)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN3(mad)

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(maxmag)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(minmag)


HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_modf_f32(float x, float* y ) {
  return __hipsycl_modf(x, y);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_modf_f64(double x, double* y) {
  return __hipsycl_modf(x, y);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(nextafter)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(powr)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_pown_f32(float x, __hipsycl_int32 y) {
  return __hipsycl_pown(x, y);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_pown_f64(double x,
                                                    __hipsycl_int64 y) {
  return __hipsycl_pown(x, y);
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN2(remainder)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(rint)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_rootn_f32(float x,
                                                    __hipsycl_int32 y) {
  return __hipsycl_rootn(x, static_cast<float>(y));
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_rootn_f64(double x,
                                                     __hipsycl_int64 y) {
  return __hipsycl_rootn(x, static_cast<double>(y));
}

HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(round)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(rsqrt)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sqrt)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sin)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sinh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(sinpi)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tan)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(tanh)
HIPSYCL_SSCP_MAP_HOST_FLOAT_BUILTIN(trunc)
