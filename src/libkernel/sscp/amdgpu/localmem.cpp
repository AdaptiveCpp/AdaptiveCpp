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



unsigned __llvm_amdgcn_groupstaticsize() __asm("llvm.amdgcn.groupstaticsize");
inline static __attribute__((address_space(3))) void* __to_local(unsigned x) { return (__attribute__((address_space(3))) void*)x; }

__attribute__((address_space(3))) void* __acpp_sscp_get_dynamic_local_memory() {
  return __to_local(__llvm_amdgcn_groupstaticsize());
}
