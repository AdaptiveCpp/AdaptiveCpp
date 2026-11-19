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
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

extern "C" [[clang::convergent]] void __acpp_cbs_barrier();
extern "C" [[clang::convergent]] void __acpp_cbs_sub_group_barrier();

__attribute__((always_inline)) void
__acpp_cpu_mem_fence(__acpp_sscp_memory_scope fence_scope,
                     __acpp_sscp_memory_order order) {
  // FIXME!
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order order) {

  // TODO: Correctly take into account memory order for local_barrier
  __acpp_cbs_barrier();
  if(fence_scope != __acpp_sscp_memory_scope::work_group) {
    __acpp_cpu_mem_fence(fence_scope, order);
  }
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order order) {
  // with more than one work-item per sub-group, CBS has to split the kernel here
  if (__acpp_sscp_get_subgroup_max_size() > 1)
    __acpp_cbs_sub_group_barrier();
  __acpp_cpu_mem_fence(fence_scope, order);
}


HIPSYCL_SSCP_BUILTIN
void __acpp_sscp_memory_fence(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order order) {
  int fence_order = __ATOMIC_RELAXED;

  if(order == __acpp_sscp_memory_order::acq_rel)
    fence_order = __ATOMIC_ACQ_REL;
  else if(order == __acpp_sscp_memory_order::acquire)
    fence_order = __ATOMIC_ACQUIRE;
  else if(order == __acpp_sscp_memory_order::release)
    fence_order = __ATOMIC_RELEASE;
  else if(order == __acpp_sscp_memory_order::seq_cst)
    fence_order = __ATOMIC_SEQ_CST;

  __atomic_thread_fence(fence_order);
}
