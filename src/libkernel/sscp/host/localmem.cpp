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

extern "C" void* __acpp_cbs_sscp_dynamic_local_memory;

__attribute__((address_space(3))) void* __acpp_sscp_get_dynamic_local_memory() {
  return (__attribute__((address_space(3))) void*)(__acpp_cbs_sscp_dynamic_local_memory);
}
