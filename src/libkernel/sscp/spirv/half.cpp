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
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"

// TODO How does native half work in SPIR-V?
// This file currently emulates half computation in fp32.
using hipsycl::fp16::promote_to_float;

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_add(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(promote_to_float(a) + promote_to_float(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_sub(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(promote_to_float(a) - promote_to_float(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_mul(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(promote_to_float(a) * promote_to_float(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_div(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(promote_to_float(a) / promote_to_float(b));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lt(__acpp_f16 a, __acpp_f16 b) {
  return promote_to_float(a) < promote_to_float(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lte(__acpp_f16 a, __acpp_f16 b) {
  return promote_to_float(a) <= promote_to_float(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gt(__acpp_f16 a, __acpp_f16 b) {
  return promote_to_float(a) > promote_to_float(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gte(__acpp_f16 a, __acpp_f16 b) {
  return promote_to_float(a) >= promote_to_float(b);
}
