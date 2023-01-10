/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"



namespace __spv {

enum ScopeFlag : __hipsycl_uint32 {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};

enum MemorySemanticsMaskFlag : __hipsycl_uint32 {
  None = 0x0,
  Acquire = 0x2,
  Release = 0x4,
  AcquireRelease = 0x8,
  SequentiallyConsistent = 0x10,
  UniformMemory = 0x40,
  SubgroupMemory = 0x80,
  WorkgroupMemory = 0x100,
  CrossWorkgroupMemory = 0x200,
  AtomicCounterMemory = 0x400,
  ImageMemory = 0x800
};

}

__attribute__((convergent)) extern "C" void
__spirv_ControlBarrier(__spv::ScopeFlag Execution, __spv::ScopeFlag Memory,
                       __hipsycl_uint32 Semantics);

__attribute__((always_inline)) __spv::ScopeFlag
get_spirv_scope(__hipsycl_sscp_memory_scope scope) {

  if(scope == __hipsycl_sscp_memory_scope::work_item)
    return __spv::ScopeFlag::Invocation;
  else if(scope == __hipsycl_sscp_memory_scope::sub_group)
    return __spv::ScopeFlag::Subgroup;
  else if(scope == __hipsycl_sscp_memory_scope::work_group)
    return __spv::ScopeFlag::Workgroup;
  else if(scope == __hipsycl_sscp_memory_scope::device)
    return __spv::ScopeFlag::Device;
  else
    return __spv::ScopeFlag::CrossDevice;
}

__attribute__((always_inline)) __hipsycl_uint32
get_spirv_memory_semantics_from_order(__hipsycl_sscp_memory_order order) {
  if(order == __hipsycl_sscp_memory_order::seq_cst)
    return __spv::MemorySemanticsMaskFlag::SequentiallyConsistent;
  else if(order == __hipsycl_sscp_memory_order::acq_rel)
    return __spv::MemorySemanticsMaskFlag::AcquireRelease;
  else if(order == __hipsycl_sscp_memory_order::release)
    return __spv::MemorySemanticsMaskFlag::Release;
  else if(order == __hipsycl_sscp_memory_order::acquire)
    return __spv::MemorySemanticsMaskFlag::Acquire;
  else // Relaxed TODO: How to best map relaxed to SPIR-V?
    return __spv::MemorySemanticsMaskFlag::SequentiallyConsistent;
}


HIPSYCL_SSCP_CONVERGENT_BUILTIN void
    __hipsycl_sscp_work_group_barrier(__hipsycl_sscp_memory_scope fence_scope,
                                  __hipsycl_sscp_memory_order mem_order) {

  __hipsycl_uint32 flags = get_spirv_memory_semantics_from_order(mem_order);

  if(fence_scope == __hipsycl_sscp_memory_scope::sub_group)
    flags |= __spv::MemorySemanticsMaskFlag::SubgroupMemory;
  else if(fence_scope == __hipsycl_sscp_memory_scope::work_group)
    flags |= __spv::MemorySemanticsMaskFlag::WorkgroupMemory;
  else if(fence_scope == __hipsycl_sscp_memory_scope::device)
    flags |= __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory;

  __spv::ScopeFlag mem_fence_scope = get_spirv_scope(fence_scope);

  __spirv_ControlBarrier(__spv::ScopeFlag::Workgroup, mem_fence_scope, flags);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
    __hipsycl_sscp_sub_group_barrier(__hipsycl_sscp_memory_scope fence_scope,
                                  __hipsycl_sscp_memory_order mem_order) {
  __hipsycl_uint32 flags = get_spirv_memory_semantics_from_order(mem_order);

  if(fence_scope == __hipsycl_sscp_memory_scope::sub_group)
    flags |= __spv::MemorySemanticsMaskFlag::SubgroupMemory;
  else if(fence_scope == __hipsycl_sscp_memory_scope::work_group)
    flags |= __spv::MemorySemanticsMaskFlag::WorkgroupMemory;
  else if(fence_scope == __hipsycl_sscp_memory_scope::device)
    flags |= __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory;

  __spv::ScopeFlag mem_fence_scope = get_spirv_scope(fence_scope);

  __spirv_ControlBarrier(__spv::ScopeFlag::Subgroup, mem_fence_scope, flags);
  
}


