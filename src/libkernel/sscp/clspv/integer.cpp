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

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_mul24_s32(__acpp_int32 a,
                                                        __acpp_int32 b) {
  // clspv uses 32-bits to implement `mul24()` so preempt this
  return a * b;
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_mul24_u32(__acpp_uint32 a,
                                                         __acpp_uint32 b) {
  // clspv uses 32-bits to implement `mul24()` so preempt this
  return a * b;
}

__acpp_uint8 ctz(__acpp_uint8);
__acpp_uint16 ctz(__acpp_uint16);
__acpp_uint32 ctz(__acpp_uint32);
__acpp_uint64 ctz(__acpp_uint64);

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_ctz_u32(__acpp_uint32 a) {
  return ctz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_ctz_u64(__acpp_uint64 a) {
  return ctz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_ctz_u8(__acpp_uint8 a) {
  return ctz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_ctz_u16(__acpp_uint16 a) {
  return ctz(a);
}

__acpp_uint8 clz(__acpp_uint8);
__acpp_uint16 clz(__acpp_uint16);
__acpp_uint32 clz(__acpp_uint32);
__acpp_uint64 clz(__acpp_uint64);

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_clz_u32(__acpp_uint32 a) {
  return clz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_clz_u64(__acpp_uint64 a) {
  return clz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_clz_u8(__acpp_uint8 a) {
  return clz(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_clz_u16(__acpp_uint16 a) {
  return clz(a);
}

__acpp_uint32 popcount(__acpp_uint32);
__acpp_uint64 popcount(__acpp_uint64);

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_popcount_u32(__acpp_uint32 a) {
  return popcount(a);
}
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_popcount_u64(__acpp_uint64 a) {
  return popcount(a);
}
