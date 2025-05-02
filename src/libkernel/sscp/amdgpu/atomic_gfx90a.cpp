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
#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"


HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_unsafe_atomic_fetch_add_f32(
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
  __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  return __builtin_amdgcn_global_atomic_fadd_f64(ptr, x);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_unsafe_atomic_fetch_add_f64(
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
  __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  return __builtin_amdgcn_global_atomic_fadd_f32(ptr, x);
}

