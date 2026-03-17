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
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"


void sub_group_barrier(unsigned) __attribute__((convergent));
void barrier(unsigned) __attribute__((convergent));
void mem_fence(unsigned) __attribute__((convergent));

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope scope,
                               __acpp_sscp_memory_order) {
  barrier(3 /* CL_LOCAL_MEM_FENCE | CL_GLOBAL_MEM_FENCE */);
}

HIPSYCL_SSCP_BUILTIN
void __acpp_sscp_memory_fence(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order) {
  mem_fence(3 /* CL_LOCAL_MEM_FENCE | CL_GLOBAL_MEM_FENCE */);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order) {
  sub_group_barrier(3 /* CL_LOCAL_MEM_FENCE | CL_GLOBAL_MEM_FENCE */);
}
