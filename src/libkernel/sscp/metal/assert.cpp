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

#include "hipSYCL/sycl/libkernel/sscp/builtins/assert.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_symbol_assert(const char* s);

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_assert_fail(const char *assertion,
                                                  const char *file,
                                                  __acpp_uint32 line,
                                                  const char *function) {
  __acpp_sscp_metal_symbol_assert("__builtin_trap()");
}

HIPSYCL_SSCP_BUILTIN void
__acpp_sscp_glibcxx_assert_fail(const char *file, __acpp_int32 line,
                                const char *function, const char *assertion) {
  __acpp_sscp_metal_symbol_assert("__builtin_trap()");
}
