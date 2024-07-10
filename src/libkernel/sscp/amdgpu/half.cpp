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

using hipsycl::fp16::as_native_float16;

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_add(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(as_native_float16(a) + as_native_float16(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_sub(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(as_native_float16(a) - as_native_float16(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_mul(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(as_native_float16(a) * as_native_float16(b));
}

HIPSYCL_SSCP_BUILTIN __acpp_f16 __acpp_sscp_half_div(__acpp_f16 a,
                                                     __acpp_f16 b) {
  return hipsycl::fp16::create(as_native_float16(a) / as_native_float16(b));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lt(__acpp_f16 a, __acpp_f16 b) {
  return as_native_float16(a) < as_native_float16(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_lte(__acpp_f16 a, __acpp_f16 b) {
  return as_native_float16(a) <= as_native_float16(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gt(__acpp_f16 a, __acpp_f16 b) {
  return as_native_float16(a) > as_native_float16(b);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_half_gte(__acpp_f16 a, __acpp_f16 b) {
  return as_native_float16(a) >= as_native_float16(b);
}
