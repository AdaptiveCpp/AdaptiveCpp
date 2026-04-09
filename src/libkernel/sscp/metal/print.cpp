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

#include "hipSYCL/sycl/libkernel/sscp/builtins/print.hpp"

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_symbol_assert(const char* s);

void __acpp_sscp_print(const char* msg) {
  //TODO: use <metal_logging>
  __acpp_sscp_metal_symbol_assert("__builtin_trap()");
}
