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
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"

// Compiler will set a proper size for this array based on local memory size
extern "C" __attribute__((address_space(3))) int __acpp_sscp_spirv_dynamic_local_mem [];

HIPSYCL_SSCP_BUILTIN __attribute__((address_space(3))) void *
__acpp_sscp_get_dynamic_local_memory() {
  return __acpp_sscp_spirv_dynamic_local_mem;
}
