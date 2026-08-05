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
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"

#define PI 3.14159265358979323846

float acos(float);
double acos(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acos_f32(float x) { return acos(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acos_f64(double x) { return acos(x); }

float acosh(float);
double acosh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acosh_f32(float x) { return acosh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acosh_f64(double x) { return acosh(x); }

float acospi(float);
double acospi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acospi_f32(float x) { return acospi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acospi_f64(double x) {
  return acospi(x);
}

float asin(float);
double asin(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asin_f32(float x) { return asin(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asin_f64(double x) { return asin(x); }

float asinh(float);
double asinh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinh_f32(float x) { return asinh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinh_f64(double x) { return asinh(x); }

float asinpi(float);
double asinpi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinpi_f32(float x) { return asinpi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinpi_f64(double x) {
  return asinpi(x);
}

float atan(float);
double atan(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan_f32(float x) { return atan(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan_f64(double x) { return atan(x); }

float atan2(float, float);
double atan2(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2_f32(float x, float y) {
  return atan2(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2_f64(double x, double y) {
  return atan2(x, y);
}

float atanh(float);
double atanh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanh_f32(float x) { return atanh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanh_f64(double x) { return atanh(x); }

float atanpi(float);
double atanpi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanpi_f32(float x) { return atanpi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanpi_f64(double x) {
  return atanpi(x);
}

float atan2pi(float, float);
double atan2pi(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2pi_f32(float x, float y) {
  return atan2pi(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2pi_f64(double x, double y) {
  return atan2pi(x, y);
}

float cbrt(float);
double cbrt(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cbrt_f32(float x) { return cbrt(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cbrt_f64(double x) { return cbrt(x); }

float ceil(float);
double ceil(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ceil_f32(float x) { return ceil(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ceil_f64(double x) { return ceil(x); }

float copysign(float, float);
double copysign(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_copysign_f32(float x, float y) {
  return copysign(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_copysign_f64(double x, double y) {
  return copysign(x, y);
}

float cos(float);
double cos(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cos_f32(float x) { return cos(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cos_f64(double x) { return cos(x); }

float cosh(float);
double cosh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cosh_f32(float x) { return cosh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cosh_f64(double x) { return cosh(x); }

float cospi(float);
double cospi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cospi_f32(float x) { return cospi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cospi_f64(double x) { return cospi(x); }

float erfc(float);
double erfc(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_erfc_f32(float x) { return erfc(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_erfc_f64(double x) { return erfc(x); }

float erf(float);
double erf(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_erf_f32(float x) { return erf(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_erf_f64(double x) { return erf(x); }

float exp(float);
double exp(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp_f32(float x) { return exp(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp_f64(double x) { return exp(x); }

float exp2(float);
double exp2(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp2_f32(float x) { return exp2(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp2_f64(double x) { return exp2(x); }

float exp10(float);
double exp10(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp10_f32(float x) { return exp10(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp10_f64(double x) { return exp10(x); }

float expm1(float);
double expm1(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_expm1_f32(float x) { return expm1(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_expm1_f64(double x) { return expm1(x); }

float fabs(float);
double fabs(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fabs_f32(float x) { return fabs(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fabs_f64(double x) { return fabs(x); }

float fdim(float, float);
double fdim(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fdim_f32(float x, float y) {
  return fdim(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fdim_f64(double x, double y) {
  return fdim(x, y);
}

float floor(float);
double floor(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_floor_f32(float x) { return floor(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_floor_f64(double x) { return floor(x); }

float fma(float, float, float);
double fma(double, double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fma_f32(float x, float y, float z) {
  return fma(x, y, z);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fma_f64(double x, double y, double z) {
  return fma(x, y, z);
}

float fmax(float, float);
double fmax(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmax_f32(float x, float y) {
  return fmax(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmax_f64(double x, double y) {
  return fmax(x, y);
}

float fmin(float, float);
double fmin(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmin_f32(float x, float y) {
  return fmin(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmin_f64(double x, double y) {
  return fmin(x, y);
}

float fmod(float, float);
double fmod(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmod_f32(float x, float y) {
  return fmod(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmod_f64(double x, double y) {
  return fmod(x, y);
}

float fract(float, float *);
double fract(double, double *);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fract_f32(float x, float *y) {
  return fract(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double x, double *y) {
  return fract(x, y);
}

float frexp(float, __acpp_int32 *);
double frexp(double, __acpp_int32 *);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_frexp_f32(float x, __acpp_int32 *y) {
  return frexp(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double x, __acpp_int32 *y) {
  return frexp(x, y);
}

float hypot(float, float);
double hypot(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_hypot_f32(float x, float y) {
  return hypot(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_hypot_f64(double x, double y) {
  return hypot(x, y);
}

__acpp_int32 ilogb(float);
__acpp_int32 ilogb(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_ilogb_f32(float x) { return ilogb(x); }
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_ilogb_f64(double x) { return ilogb(x); }

float ldexp(float, __acpp_int32);
double ldexp(double, __acpp_int32);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ldexp_f32(float x, __acpp_int32 y) {
  return ldexp(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double x, __acpp_int32 y) {
  return ldexp(x, y);
}

float lgamma(float);
double lgamma(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_f32(float x) { return lgamma(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_f64(double x) {
  return lgamma(x);
}

float lgamma_r(float, __acpp_int32 *);
double lgamma_r(double, __acpp_int32 *);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_r_f32(float x, __acpp_int32 *y) {
  return lgamma_r(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double x,
                                                     __acpp_int32 *y) {
  return lgamma_r(x, y);
}

float log(float);
double log(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log_f32(float x) { return log(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log_f64(double x) { return log(x); }

float log2(float);
double log2(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log2_f32(float x) { return log2(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log2_f64(double x) { return log2(x); }

float log10(float);
double log10(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log10_f32(float x) { return log10(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log10_f64(double x) { return log10(x); }

float log1p(float);
double log1p(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log1p_f32(float x) { return log1p(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log1p_f64(double x) { return log1p(x); }

float logb(float);
double logb(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_logb_f32(float x) { return logb(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_logb_f64(double x) { return logb(x); }

float mad(float, float, float);
double mad(double, double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_mad_f32(float x, float y, float z) {
  return mad(x, y, z);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_mad_f64(double x, double y, double z) {
  return mad(x, y, z);
}

float maxmag(float, float);
double maxmag(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_maxmag_f32(float x, float y) {
  return maxmag(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_maxmag_f64(double x, double y) {
  return maxmag(x, y);
}

float minmag(float, float);
double minmag(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_minmag_f32(float x, float y) {
  return minmag(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_minmag_f64(double x, double y) {
  return minmag(x, y);
}

float modf(float, float *);
double modf(double, double *);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_modf_f32(float x, float *y) {
  return modf(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double x, double *y) {
  return modf(x, y);
}

float nextafter(float, float);
double nextafter(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_nextafter_f32(float x, float y) {
  return nextafter(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_nextafter_f64(double x, double y) {
  return nextafter(x, y);
}

float pow(float, float);
double pow(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pow_f32(float x, float y) {
  return pow(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pow_f64(double x, double y) {
  return pow(x, y);
}

float powr(float, float);
double powr(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_powr_f32(float x, float y) {
  return powr(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_powr_f64(double x, double y) {
  return powr(x, y);
}

float pown(float, __acpp_int32);
double pown(double, __acpp_int32);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pown_f32(float x, __acpp_int32 y) {
  return pown(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double x, __acpp_int32 y) {
  return pown(x, y);
}

float remainder(float, float);
double remainder(double, double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_remainder_f32(float x, float y) {
  return remainder(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_remainder_f64(double x, double y) {
  return remainder(x, y);
}

float rint(float);
double rint(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rint_f32(float x) { return rint(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rint_f64(double x) { return rint(x); }

float rootn(float, __acpp_int32);
double rootn(double, __acpp_int32);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rootn_f32(float x, __acpp_int32 y) {
  return rootn(x, y);
}
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double x, __acpp_int32 y) {
  return rootn(x, y);
}

float round(float);
double round(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_round_f32(float x) { return round(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_round_f64(double x) { return round(x); }

float rsqrt(float);
double rsqrt(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rsqrt_f32(float x) { return rsqrt(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rsqrt_f64(double x) { return rsqrt(x); }

float sin(float);
double sin(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sin_f32(float x) { return sin(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sin_f64(double x) { return sin(x); }

float sincos(float, float *);
double sincos(double, double *);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_sincos_f32(float x, float *sinval,
                                                 float *cosval) {
  *sinval = sincos(x, cosval);
}
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_sincos_f64(double x, double *sinval,
                                                 double *cosval) {
  *sinval = sincos(x, cosval);
}

float sinh(float);
double sinh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sinh_f32(float x) { return sinh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sinh_f64(double x) { return sinh(x); }

float sinpi(float);
double sinpi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sinpi_f32(float x) { return sinpi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sinpi_f64(double x) { return sinpi(x); }

float sqrt(float);
double sqrt(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sqrt_f32(float x) { return sqrt(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sqrt_f64(double x) { return sqrt(x); }

float tan(float);
double tan(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tan_f32(float x) { return tan(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tan_f64(double x) { return tan(x); }

float tanpi(float);
double tanpi(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tanpi_f32(float x) { return tanpi(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tanpi_f64(double x) { return tanpi(x); }

float tanh(float);
double tanh(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tanh_f32(float x) { return tanh(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tanh_f64(double x) { return tanh(x); }

float tgamma(float);
double tgamma(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tgamma_f32(float x) { return tgamma(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tgamma_f64(double x) {
  return tgamma(x);
}

float trunc(float);
double trunc(double);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_trunc_f32(float x) { return trunc(x); }
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_trunc_f64(double x) { return trunc(x); }
