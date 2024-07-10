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
#include "hipSYCL/sycl/libkernel/sscp/builtins/spirv/spirv_common.hpp"


__attribute__((convergent)) extern "C" void
__spirv_ControlBarrier(__spv::ScopeFlag Execution, __spv::ScopeFlag Memory,
                       __acpp_uint32 Semantics);

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order mem_order) {

  __acpp_uint32 flags = get_spirv_memory_semantics(mem_order);

  if(fence_scope == __acpp_sscp_memory_scope::sub_group)
    flags |= __spv::MemorySemanticsMaskFlag::SubgroupMemory;
  else if(fence_scope == __acpp_sscp_memory_scope::work_group)
    flags |= __spv::MemorySemanticsMaskFlag::WorkgroupMemory;
  else if(fence_scope == __acpp_sscp_memory_scope::device)
    flags |= __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory;

  __spv::ScopeFlag mem_fence_scope = get_spirv_scope(fence_scope);

  __spirv_ControlBarrier(__spv::ScopeFlag::Workgroup, mem_fence_scope, flags);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order mem_order) {
  __acpp_uint32 flags = get_spirv_memory_semantics(mem_order);

  if(fence_scope == __acpp_sscp_memory_scope::sub_group)
    flags |= __spv::MemorySemanticsMaskFlag::SubgroupMemory;
  else if(fence_scope == __acpp_sscp_memory_scope::work_group)
    flags |= __spv::MemorySemanticsMaskFlag::WorkgroupMemory;
  else if(fence_scope == __acpp_sscp_memory_scope::device)
    flags |= __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory;

  __spv::ScopeFlag mem_fence_scope = get_spirv_scope(fence_scope);

  __spirv_ControlBarrier(__spv::ScopeFlag::Subgroup, mem_fence_scope, flags);
}
