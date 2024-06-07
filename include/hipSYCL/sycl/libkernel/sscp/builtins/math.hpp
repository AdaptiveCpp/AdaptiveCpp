/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_MATH_BUILTINS_HPP
#define HIPSYCL_SSCP_MATH_BUILTINS_HPP

#include "builtin_config.hpp"


HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acos_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acos_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acosh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acosh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acospi_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acospi_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_acos_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_acos_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asin_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asin_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_asinpi_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_asinpi_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atanpi_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atanpi_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_atan2pi_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_atan2pi_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cbrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cbrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ceil_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ceil_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_copysign_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_copysign_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cos_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cos_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cosh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cosh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_cospi_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_cospi_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_erfc_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_erfc_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_erf_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_erf_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp2_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp2_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_exp10_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_exp10_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_expm1_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_expm1_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fabs_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fabs_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fdim_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fdim_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_floor_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_floor_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fma_f32(float, float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fma_f64(double, double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmax_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmax_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmin_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmin_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fmod_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fmod_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_fract_f32(float, float*);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_fract_f64(double, double*);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_frexp_f32(float, __acpp_int32*);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_frexp_f64(double, __acpp_int64*);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_hypot_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_hypot_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ilogb_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ilogb_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_ldexp_f32(float, __acpp_int32);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_ldexp_f64(double, __acpp_int64);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_lgamma_r_f32(float, __acpp_int32*);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_lgamma_r_f64(double, __acpp_int64*);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log2_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log2_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log10_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log10_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_log1p_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_log1p_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_logb_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_logb_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_mad_f32(float, float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_mad_f64(double, double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_maxmag_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_maxmag_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_minmag_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_minmag_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_modf_f32(float, float*);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_modf_f64(double, double*);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_nextafter_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_nextafter_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pow_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pow_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_powr_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_powr_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_pown_f32(float, __acpp_int32);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_pown_f64(double, __acpp_int64);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_remainder_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_remainder_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rint_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rint_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rootn_f32(float, __acpp_int32);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rootn_f64(double, __acpp_int64);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_round_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_round_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rsqrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rsqrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_round_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_round_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_rsqrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_rsqrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sin_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sin_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sinh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sinh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sinpi_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sinpi_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_sqrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_sqrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tan_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tan_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tanh_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tanh_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_tgamma_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_tgamma_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_trunc_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_trunc_f64(double);


#endif
