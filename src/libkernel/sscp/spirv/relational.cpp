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

#define HIPSYCL_DECLARE_SSCP_SPIRV_BUILTIN(dispatched_name)                    \
  int __spirv_##dispatched_name(float);                                        \
  int __spirv_##dispatched_name(double);

#define HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(builtin_name,                \
                                                  dispatched_name)             \
  HIPSYCL_DECLARE_SSCP_SPIRV_BUILTIN(dispatched_name)                          \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##builtin_name##_f32(    \
      float x) {                                                               \
    return __spirv_##dispatched_name(x);                                       \
  }                                                                            \
  HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_##builtin_name##_f64(    \
      double x) {                                                              \
    return __spirv_##dispatched_name(x);                                       \
  }

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isnan, IsNan)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isinf, IsInf)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isfinite, IsFinite)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(isnormal, IsNormal)

HIPSYCL_SSCP_MAP_BUILTIN_TO_SPIRV_BUILTIN(signbit, SignBitSet)
