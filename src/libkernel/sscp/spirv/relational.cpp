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

#include "hipSYCL/sycl/libkernel/sscp/builtins/relational.hpp"

#define HIPSYCL_DECLARE_SSCP_SPIRV_BUILTIN(dispatched_name)                    \
  int __spirv_##dispatched_name(float);                                        \
  int __spirv_##dispatched_name(double);

#define HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(builtin_name,                \
                                                  dispatched_name)             \
  HIPSYCL_DECLARE_SSCP_SPIRV_BUILTIN(dispatched_name)                          \
  HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_##builtin_name##_f32(    \
      float x) {                                                               \
    return __spirv_##dispatched_name(x);                                       \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_##builtin_name##_f64(    \
      double x) {                                                              \
    return __spirv_##dispatched_name(x);                                       \
  }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isnan, IsNan)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isinf, IsInf)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isfinite, IsFinite)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isnormal, IsNormal)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(signbit, SignBitSet)
