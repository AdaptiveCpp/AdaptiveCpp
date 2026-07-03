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
#include "hipSYCL/sycl/libkernel/sscp/builtins/native.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/math.hpp"

float native_cos(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_cos_f32(float x) {
  return native_cos(x);
}

float native_divide(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_divide_f32(float x, float y) {
  return native_divide(x);
}

float native_exp(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp_f32(float x) {
  return native_exp(x);
}

float native_exp2(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp2_f32(float x) {
  return native_exp2(x);
}

float native_exp10(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_exp10_f32(float x) {
  return native_exp10(x);
}

float native_log(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log_f32(float x) {
  return native_log(x);
}

float native_log2(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log2_f32(float x) {
  return native_log2(x);
}

float native_log10(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_log10_f32(float x) {
  return native_log10(x);
}

float native_powr(float, float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_powr_f32(float x, float y) {
  return native_powr(x, y);
}

float native_recip(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_recip_f32(float x) {
  return native_recip(x);
}

float native_rsqrt(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_rsqrt_f32(float x) {
  return native_rsqrt(x);
}

float native_sin(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sin_f32(float x) {
  return native_sin(x);
}

float native_sqrt(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_sqrt_f32(float x) {
  return native_sqrt(x);
}

float native_tan(float);
HIPSYCL_SSCP_BUILTIN float __acpp_sscp_native_tan_f32(float x) {
  return native_tan(x);
}
