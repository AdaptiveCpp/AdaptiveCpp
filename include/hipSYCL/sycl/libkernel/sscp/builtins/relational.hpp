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
#include "builtin_config.hpp"

#ifndef HIPSYCL_SSCP_RELATIONAL_BUILTINS_HPP
#define HIPSYCL_SSCP_RELATIONAL_BUILTINS_HPP

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnan_f32(float);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnan_f64(double);

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isinf_f32(float);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isinf_f64(double);

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f32(float);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f64(double);

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f32(float);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f64(double);

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f32(float);
HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_signbit_f64(double);

#endif
