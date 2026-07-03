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
#include "hipSYCL/sycl/libkernel/sscp/builtins/relational.hpp"

__acpp_int32 isnan(float);
__acpp_int32 isnan(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnan_f32(float x) {
  return isnan(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnan_f64(double x) {
  return isnan(x);
}

__acpp_int32 isinf(float);
__acpp_int32 isinf(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isinf_f32(float x) {
  return isinf(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isinf_f64(double x) {
  return isinf(x);
}

__acpp_int32 isfinite(float);
__acpp_int32 isfinite(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f32(float x) {
  return isfinite(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f64(double x) {
  return isfinite(x);
}

__acpp_int32 isnormal(float);
__acpp_int32 isnormal(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f32(float x) {
  return isnormal(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f64(double x) {
  return isnormal(x);
}

__acpp_int32 signbit(float);
__acpp_int32 signbit(double);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f32(float x) {
  return signbit(x);
}
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f64(double x) {
  return signbit(x);
}
