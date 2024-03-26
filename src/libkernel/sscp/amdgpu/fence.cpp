/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/sycl/libkernel/sscp/builtins/fence.hpp"

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

extern "C" void
__atomic_work_item_fence(unsigned mem_fence_flags, amdgpu_memory_order, amdgpu_memory_scope);

__attribute__((always_inline)) amdgpu_memory_order
__hipsycl_amdgpu_get_mem_order(__hipsycl_sscp_memory_order order) {

  if(order == __hipsycl_sscp_memory_order::relaxed)
    return relaxed;
  else if(order == __hipsycl_sscp_memory_order::acquire)
    return acquire;
  else if(order == __hipsycl_sscp_memory_order::release)
    return release;
  else if(order == __hipsycl_sscp_memory_order::acq_rel)
    return acq_rel;
  else
    return seq_cst;
}

__attribute__((always_inline)) amdgpu_memory_scope
__hipsycl_amdgpu_get_mem_scope(__hipsycl_sscp_memory_scope scope) {

  if(scope == __hipsycl_sscp_memory_scope::work_item)
    return work_item;
  else if(scope == __hipsycl_sscp_memory_scope::sub_group)
    return sub_group;
  else if(scope == __hipsycl_sscp_memory_scope::work_group)
    return work_group;
  else if(scope == __hipsycl_sscp_memory_scope::device)
    return device;
  else
    return all_svm_devices;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__hipsycl_sscp_fence(__hipsycl_sscp_memory_order order,
                     __hipsycl_sscp_memory_scope scope) {

  auto mem_order = __hipsycl_amdgpu_get_mem_order(order);
  auto mem_scope = __hipsycl_amdgpu_get_mem_scope(scope);
  
  __atomic_work_item_fence(0, mem_order, mem_scope);
}
