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

#ifndef HIPSYCL_ATOMIC_FENCE_HPP
#define HIPSYCL_ATOMIC_FENCE_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "memory.hpp"

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/sycl/libkernel/sscp/builtins/fence.hpp"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {

HIPSYCL_KERNEL_TARGET
inline void atomic_fence(memory_order order, memory_scope scope)
{
  __hipsycl_if_target_hiplike(
    if (order == memory_order::relaxed ||
        scope == memory_scope::work_item)
      ;
    else if(scope <= memory_scope::work_group)
      __threadfence_block();
    else
      __threadfence();
  );

  __hipsycl_if_target_spirv(
    __hipsycl_uint32 flags;
    __spv::Scope fence_scope;

    if(order == memory_order::relaxed)
      flags = __spv::MemorySemanticsMask::None;
    else if(order == memory_order::acquire)
      flags = __spv::MemorySemanticsMask::Acquire;
    else if(order == memory_order::release)
      flags = __spv::MemorySemanticsMask::Release;
    else if(order == memory_order::acq_rel)
      flags = __spv::MemorySemanticsMask::AcquireRelease;
    else if(order == memory_order::seq_cst)
      flags = __spv::MemorySemanticsMask::SequentiallyConsistent;

    if(scope >= memory_scope::sub_group)
      flags |= __spv::MemorySemanticsMask::SubgroupMemory;
    if(scope >= memory_scope::work_group)
      flags |= __spv::MemorySemanticsMask::WorkgroupMemory;
    if(scope >= memory_scope::device)
      flags |= __spv::MemorySemanticsMask::CrossWorkgroupMemory;

    if(scope == memory_scope::work_item)
      fence_scope = __spv::ScopeFlag::Invocation;
    else if(scope == memory_scope::sub_group)
      fence_scope = __spv::ScopeFlag::Subgroup;
    else if(scope == memory_scope::work_group)
      fence_scope = __spv::ScopeFlag::Workgroup;
    else if(scope == memory_scope::device)
      fence_scope = __spv::ScopeFlag::Device;
    else if(scope == memory_scope::system)
      fence_scope = __spv::ScopeFlag::CrossDevice;

    __spirv_MemoryBarrier(fence_scope, flags);
  );

  __hipsycl_if_target_sscp(
    __hipsycl_sscp_fence(order, scope);
  );

  // TODO What about CPU?
  // Empty __hipsycl_if_target_* breaks at compile time w/ nvc++ 22.7 or
  // older, so comment out that statement for now.
  //__hipsycl_if_target_host(/* todo */);
}

} // namespace detail

HIPSYCL_KERNEL_TARGET
static inline void atomic_fence(memory_order order, memory_scope scope)
{
  detail::atomic_fence(order, scope);
}

} // namespace sycl
} // namespace hipsycl

#endif
