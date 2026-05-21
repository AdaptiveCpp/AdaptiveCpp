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

#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_half_cmp(const char* name, u16 x, u16 y);

#define ACPP_SSCP_MAP_METAL_HALF_CMP(op, symbol) \
  HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_##op(u16 x, u16 y) { \
    return __acpp_sscp_metal_half_cmp("as_type<half>(%s) " #symbol " as_type<half>(%s)", x, y); \
  }

ACPP_SSCP_MAP_METAL_HALF_CMP(lt, <)
ACPP_SSCP_MAP_METAL_HALF_CMP(lte, <=)
ACPP_SSCP_MAP_METAL_HALF_CMP(gt, >)
ACPP_SSCP_MAP_METAL_HALF_CMP(gte, >=)
