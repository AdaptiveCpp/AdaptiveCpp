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

template <typename... Args>
extern int __spirv_ocl_printf(const char *Format, Args... args);

void __acpp_sscp_print(const char* msg) {
  __spirv_ocl_printf(msg);
}
