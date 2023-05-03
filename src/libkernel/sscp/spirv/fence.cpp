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
#include "hipSYCL/sycl/libkernel/sscp/builtins/spirv/spirv_common.hpp"

__attribute__((convergent)) extern "C" void
__spirv_MemoryBarrier(__spv::ScopeFlag Memory, __hipsycl_uint32 Semantics);

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__hipsycl_sscp_fence(__hipsycl_sscp_memory_order order,
                     __hipsycl_sscp_memory_scope scope) {

  __hipsycl_uint32 flags = get_spirv_memory_semantics(order);

  if(scope >= __hipsycl_sscp_memory_scope::sub_group)
    flags |= __spv::MemorySemanticsMaskFlag::SubgroupMemory;
  if(scope >= __hipsycl_sscp_memory_scope::work_group)
    flags |= __spv::MemorySemanticsMaskFlag::WorkgroupMemory;
  if(scope >= __hipsycl_sscp_memory_scope::device)
    flags |= __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory;

  __spv::ScopeFlag fence_scope = get_spirv_scope(scope);

  __spirv_MemoryBarrier(fence_scope, flags);
}
