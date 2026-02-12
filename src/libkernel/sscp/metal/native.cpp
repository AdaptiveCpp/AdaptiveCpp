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

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32(const char* name, f32 x);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_math_f32_f32_f32(const char* name, f32 x, f32 y);

#define ACPP_SSCP_MAP_METAL_NATIVE_F1(name, metal_name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_native_##name##_f32(f32 x) { \
    return __acpp_sscp_metal_math_f32_f32(metal_name, x); \
  }

#define ACPP_SSCP_MAP_METAL_NATIVE_F2(name, metal_name) \
  HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_native_##name##_f32(f32 x, f32 y) { \
    return __acpp_sscp_metal_math_f32_f32_f32(metal_name, x, y); \
  }

ACPP_SSCP_MAP_METAL_NATIVE_F1(cos,    "fast::cos")
ACPP_SSCP_MAP_METAL_NATIVE_F1(sin,    "fast::sin")
ACPP_SSCP_MAP_METAL_NATIVE_F1(tan,    "fast::tan")
ACPP_SSCP_MAP_METAL_NATIVE_F1(exp,    "fast::exp")
ACPP_SSCP_MAP_METAL_NATIVE_F1(exp2,   "fast::exp2")
ACPP_SSCP_MAP_METAL_NATIVE_F1(exp10,  "fast::exp10")
ACPP_SSCP_MAP_METAL_NATIVE_F1(log,    "fast::log")
ACPP_SSCP_MAP_METAL_NATIVE_F1(log2,   "fast::log2")
ACPP_SSCP_MAP_METAL_NATIVE_F1(log10,  "fast::log10")
ACPP_SSCP_MAP_METAL_NATIVE_F1(rsqrt,  "fast::rsqrt")
ACPP_SSCP_MAP_METAL_NATIVE_F1(sqrt,   "sqrt")
ACPP_SSCP_MAP_METAL_NATIVE_F2(powr,   "fast::powr")
ACPP_SSCP_MAP_METAL_NATIVE_F2(divide, "fast::divide")

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_native_recip_f32(f32 x) {
  return __acpp_sscp_metal_math_f32_f32("fast::recip", x);
}
