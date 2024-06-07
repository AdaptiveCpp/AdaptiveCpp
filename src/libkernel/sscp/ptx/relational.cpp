/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/relational.hpp"

#define HIPSYCL_SSCP_MAP_PTX_REL_BUILTIN(name)                                 \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##name##_f32(float x) {        \
    return __nv_##name##f(x);                                                  \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##name##_f64(double x) {       \
    return __nv_##name##d(x);                                                  \
  }

HIPSYCL_SSCP_MAP_PTX_REL_BUILTIN(isnan)

HIPSYCL_SSCP_MAP_PTX_REL_BUILTIN(isinf)

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f32(float x) {
  return __nv_finitef(x);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isfinite_f64(double x) {
  return __nv_isfinited(x);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f32(float x) {
  return __builtin_isnormal(x);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_isnormal_f64(double x) {
  return __builtin_isnormal(x);
}

HIPSYCL_SSCP_MAP_PTX_REL_BUILTIN(signbit)
