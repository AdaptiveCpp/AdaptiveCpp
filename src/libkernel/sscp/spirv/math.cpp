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

#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#define PI 3.14159265358979323846

#define HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC(name)                          \
  float __spirv_ocl_##name(float);                                             \
  double __spirv_ocl_##name(double);

#define HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC2(name)                         \
  float __spirv_ocl_##name(float, float);                                      \
  double __spirv_ocl_##name(double, double);

#define HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC3(name)                         \
  float __spirv_ocl_##name(float, float, float);                               \
  double __spirv_ocl_##name(double, double, double);

#define HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(name)                        \
  HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC(name)                                \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x) {            \
    return __spirv_ocl_##name(x);                                              \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x) {          \
    return __spirv_ocl_##name(x);                                              \
  }

#define HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(name)                       \
  HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC2(name)                               \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x, float y) {   \
    return __spirv_ocl_##name(x, y);                                           \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x,            \
                                                          double y) {          \
    return __spirv_ocl_##name(x, y);                                           \
  }

#define HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN3(name)                       \
  HIPSYCL_DEFINE_SSCP_SPIRV_OCL_INTRINSIC3(name)                               \
  HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_##name##_f32(float x, float y,     \
                                                         float z) {            \
    return __spirv_ocl_##name(x, y, z);                                        \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_##name##_f64(double x, double y,  \
                                                          double z) {          \
    return __spirv_ocl_##name(x, y, z);                                        \
  }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(acos)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(acosh)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_acospi_f32(float x) { return __hipsycl_sscp_acos_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_acospi_f64(double x) { return __hipsycl_sscp_acos_f64(x) / PI; }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(asin)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(asinh)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_asinpi_f32(float x) { return __hipsycl_sscp_asin_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_asinpi_f64(double x) { return __hipsycl_sscp_asin_f64(x) / PI; }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(atan)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(atan2)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(atanh)


HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_atanpi_f32(float x) { return __hipsycl_sscp_atan_f32(x) / PI; }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_atanpi_f64(double x) { return __hipsycl_sscp_atan_f64(x) / PI; }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_atan2pi_f32(float x, float y) { return __hipsycl_sscp_atan2_f32(x, y) / PI; }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_atan2pi_f64(double x, double y) { return __hipsycl_sscp_atan2_f64(x, y) / PI; }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(cbrt)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(ceil)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(copysign)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(cos)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(cosh)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_cospi_f32(float x) { return __hipsycl_sscp_cos_f32(x * PI); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_cospi_f64(double x) { return __hipsycl_sscp_cos_f64(x * PI); }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(erf)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(erfc)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(exp)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(exp2)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(exp10)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(pow)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(expm1)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(fabs)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(fdim)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(floor)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN3(fma)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(fmax)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(fmin)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(fmod)

float __spirv_ocl_fract(float x, float* y);
double __spirv_ocl_fract(double x, double* y);
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_fract_f32(float x, float* y ) { return __spirv_ocl_fract(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_fract_f64(double x, double* y) { return __spirv_ocl_fract(x, y); }

float __spirv_ocl_frexp(float x, __hipsycl_int32* y);
double __spirv_ocl_frexp(double x, __hipsycl_int32* y);
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_frexp_f32(float x, __hipsycl_int32* y ) { return __spirv_ocl_frexp(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_frexp_f64(double x, __hipsycl_int64* y) {
  __hipsycl_int32 v;
  auto res = __spirv_ocl_frexp(x, &v);
  *y = v;
  return res;
}

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(hypot)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(ilogb)

float __spirv_ocl_ldexp(float x, __hipsycl_int32 k);
double __spirv_ocl_ldexp(double x, __hipsycl_int32 k);
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_ldexp_f32(float x,
                                                    __hipsycl_int32 k) {
  return __spirv_ocl_ldexp(x, k);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_ldexp_f64(double x,
                                                     __hipsycl_int64 k) {
  return __spirv_ocl_ldexp(x, static_cast<__hipsycl_int32>(k));
}

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(tgamma)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(lgamma)

float __spirv_ocl_lgamma_r(float x, __hipsycl_int32* y);
double __spirv_ocl_lgamma_r(double x, __hipsycl_int32* y);
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_lgamma_r_f32(float x,
                                                       __hipsycl_int32 *y) {
  return __spirv_ocl_lgamma_r(x, y);
}
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_lgamma_r_f64(double x,
                                                        __hipsycl_int64 *y) {
  __hipsycl_int32 v;
  auto res = __spirv_ocl_lgamma_r(x, &v);
  *y = v;
  return res;
}

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(log)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(log2)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(log10)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(log1p)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(logb)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN3(mad)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(maxmag)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(minmag)

float __spirv_ocl_modf(float, float*);
double __spirv_ocl_modf(double, double*);
HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_modf_f32(float x, float* y ) { return __spirv_ocl_modf(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_modf_f64(double x, double* y) { return __spirv_ocl_modf(x, y); }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(nextafter)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(powr)

float __spirv_ocl_pown(float, __hipsycl_int32);
double __spirv_ocl_pown(double, __hipsycl_int32);

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_pown_f32(float x, __hipsycl_int32 y) { return __spirv_ocl_pown(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_pown_f64(double x, __hipsycl_int64 y)
{return __spirv_ocl_pown(x, static_cast<__hipsycl_int32>(y)); }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN2(remainder)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(rint)


float __spirv_ocl_rootn(float, __hipsycl_int32);
double __spirv_ocl_rootn(double, __hipsycl_int32);

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_rootn_f32(float x, __hipsycl_int32 y) { return __spirv_ocl_rootn(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_rootn_f64(double x,
                                                     __hipsycl_int64 y) {
  return __spirv_ocl_rootn(x, static_cast<__hipsycl_int32>(y));
}

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(round)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(rsqrt)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(sqrt)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(sin)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(sinh)

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_sinpi_f32(float x) { return __hipsycl_sscp_sin_f32(x * PI); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_sinpi_f64(double x) { return __hipsycl_sscp_sin_f64(x * PI); }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(tan)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(tanh)
HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(trunc)
