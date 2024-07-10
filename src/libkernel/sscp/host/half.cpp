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

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__acpp_sscp_half_add(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_add(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__acpp_sscp_half_sub(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_sub(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__acpp_sscp_half_mul(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_mul(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__acpp_sscp_half_div(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_div(a,b);
}

HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_half_lt(hipsycl::fp16::half_storage a,
                       hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_less_than(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_half_lte(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_less_than_equal(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_half_gt(hipsycl::fp16::half_storage a,
                       hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_greater_than(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_half_gte(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_greater_than(a,b);
}
