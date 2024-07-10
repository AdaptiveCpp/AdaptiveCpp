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

enum amdgpu_memory_order {
  relaxed = __ATOMIC_RELAXED,
  acquire = __ATOMIC_ACQUIRE,
  release = __ATOMIC_RELEASE,
  acq_rel = __ATOMIC_ACQ_REL,
  seq_cst = __ATOMIC_SEQ_CST
};

enum amdgpu_memory_scope {
  work_item = __OPENCL_MEMORY_SCOPE_WORK_ITEM,
  work_group = __OPENCL_MEMORY_SCOPE_WORK_GROUP,
  device = __OPENCL_MEMORY_SCOPE_DEVICE,
  all_svm_devices = __OPENCL_MEMORY_SCOPE_ALL_SVM_DEVICES,
  sub_group = __OPENCL_MEMORY_SCOPE_SUB_GROUP
};


#define __CLK_LOCAL_MEM_FENCE    0x01


extern "C" void
__atomic_work_item_fence(unsigned mem_fence_flags, amdgpu_memory_order, amdgpu_memory_scope);

__attribute__((always_inline)) amdgpu_memory_order
__acpp_amdgpu_get_mem_order(__acpp_sscp_memory_order order) {
  if(order == __acpp_sscp_memory_order::acq_rel)
    return acq_rel;
  else if(order == __acpp_sscp_memory_order::acquire)
    return acquire;
  else if(order == __acpp_sscp_memory_order::release)
    return release;
  else if(order == __acpp_sscp_memory_order::relaxed)
    return relaxed;
  else
    return seq_cst;
}

__attribute__((always_inline))
void __acpp_amdgpu_local_barrier() {
  __atomic_work_item_fence(__CLK_LOCAL_MEM_FENCE, release, work_group);
  __builtin_amdgcn_s_barrier();
  __atomic_work_item_fence(__CLK_LOCAL_MEM_FENCE, acquire, work_group);
}

__attribute__((always_inline)) void
__acpp_amdgpu_mem_fence(__acpp_sscp_memory_scope fence_scope,
                        __acpp_sscp_memory_order order) {

  auto mem_order = __acpp_amdgpu_get_mem_order(order);

  if(fence_scope == __acpp_sscp_memory_scope::work_group) {
    __atomic_work_item_fence(0, mem_order, work_group);
  } else if(fence_scope == __acpp_sscp_memory_scope::device) {
    __atomic_work_item_fence(0, mem_order, device);
  } else if(fence_scope == __acpp_sscp_memory_scope::system) {
    __atomic_work_item_fence(0, mem_order, all_svm_devices);
  }
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order order) {

  // TODO: Correctly take into account memory order for local_barrier
  __acpp_amdgpu_local_barrier();
  if(fence_scope != __acpp_sscp_memory_scope::work_group) {
    __acpp_amdgpu_mem_fence(fence_scope, order);
  }
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order order) {

  __acpp_amdgpu_mem_fence(fence_scope, order);
}
