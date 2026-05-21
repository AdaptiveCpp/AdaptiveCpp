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

#include "hipSYCL/sycl/libkernel/sscp/builtins/integer.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_metal_integer_i32_i32_i32(const char* name, i32 x, i32 y);
HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_metal_integer_u32_u32_u32(const char* name, u32 x, u32 y);

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_mul24_s32(i32 a, i32 b) {
  return __acpp_sscp_metal_integer_i32_i32_i32("mul24", a, b);
}

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_mul24_u32(u32 a, u32 b) {
  return __acpp_sscp_metal_integer_u32_u32_u32("mul24", a, b);
}
