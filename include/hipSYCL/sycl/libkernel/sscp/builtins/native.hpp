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
#ifndef HIPSYCL_SSCP_NATIVE_BUILTINS_HPP
#define HIPSYCL_SSCP_NATIVE_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_cos_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_cos_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_divide_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_divide_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp2_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp2_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp10_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_exp10_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log2_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log2_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log10_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_log10_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_powr_f32(float, float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_powr_f64(double, double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_recip_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_recip_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_rsqrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_rsqrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sin_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_sin_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sqrt_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_sqrt_f64(double);

HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_tan_f32(float);
HIPSYCL_SSCP_BUILTIN double __acpp_sscp_native_tan_f64(double);

#endif