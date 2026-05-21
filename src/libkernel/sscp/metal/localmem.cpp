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

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN __attribute__((address_space(3))) void* __acpp_sscp_metal_symbol_local_memory(const char* s);
HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_metal_symbol_local_memory_size(const char* s);

HIPSYCL_SSCP_BUILTIN
__attribute__((address_space(3))) void* __acpp_sscp_get_dynamic_local_memory() {
  return __acpp_sscp_metal_symbol_local_memory("__acpp_sscp_metal_dynamic_local_memory");
}

HIPSYCL_SSCP_BUILTIN
u32 __acpp_sscp_get_dynamic_local_memory_size() {
  return __acpp_sscp_metal_symbol_local_memory_size("__acpp_sscp_metal_dynamic_local_memory_size");
}
