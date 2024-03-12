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
#include "hipSYCL/sycl/libkernel/sscp/builtins/native.hpp"

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_cos_f32(float x) { return __hipsycl_sscp_cos_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_cos_f64(double x) { return __hipsycl_sscp_cos_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_divide_f32(float x, float y) { return x / y; }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_divide_f64(double x, double y) { return x / y; }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_exp_f32(float x) { return __hipsycl_sscp_exp_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_exp_f64(double x) { return __hipsycl_sscp_exp_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_exp2_f32(float x) { return __hipsycl_sscp_exp2_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_exp2_f64(double x) { return __hipsycl_sscp_exp2_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_exp10_f32(float x) { return __hipsycl_sscp_exp10_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_exp10_f64(double x) { return __hipsycl_sscp_exp10_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_log_f32(float x) { return __hipsycl_sscp_log_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_log_f64(double x) { return __hipsycl_sscp_log_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_log2_f32(float x) { return __hipsycl_sscp_log2_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_log2_f64(double x) { return __hipsycl_sscp_log2_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_log10_f32(float x) { return __hipsycl_sscp_log10_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_log10_f64(double x) { return __hipsycl_sscp_log10_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_powr_f32(float x, float y) { return __hipsycl_sscp_powr_f32(x, y); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_powr_f64(double x, double y) { return __hipsycl_sscp_powr_f64(x, y); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_recip_f32(float x) { return __hipsycl_sscp_native_divide_f32(1.f, x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_recip_f64(double x) { return __hipsycl_sscp_native_divide_f64(1.f, x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_rsqrt_f32(float x) { return __hipsycl_sscp_rsqrt_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_rsqrt_f64(double x) { return __hipsycl_sscp_rsqrt_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_sin_f32(float x) { return __hipsycl_sscp_sin_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_sin_f64(double x) { return __hipsycl_sscp_sin_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_sqrt_f32(float x) { return __hipsycl_sscp_sqrt_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_sqrt_f64(double x) { return __hipsycl_sscp_sqrt_f64(x); }

HIPSYCL_SSCP_BUILTIN float __hipsycl_sscp_native_tan_f32(float x) { return __hipsycl_sscp_tan_f32(x); }
HIPSYCL_SSCP_BUILTIN double __hipsycl_sscp_native_tan_f64(double x) { return __hipsycl_sscp_tan_f64(x); }
