/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#ifndef HIPSYCL_DEVICE_BARRIER_HPP
#define HIPSYCL_DEVICE_BARRIER_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"

#if !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP &&                                \
    !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA &&                               \
    !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV
#error "This file requires a device compiler"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {

HIPSYCL_KERNEL_TARGET
void local_device_barrier(
    access::fence_space space = access::fence_space::global_and_local) {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
  __syncthreads();
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
  uint32_t flags =  0;
  
  if (space == access::fence_space::global_space) {
    flags = static_cast<uint32_t>(
                   __spv::MemorySemanticsMask::SequentiallyConsistent |
                   __spv::MemorySemanticsMask::CrossWorkgroupMemory);
  } else if (space == access::fence_space::local_space){
    flags = static_cast<uint32_t>(
                         __spv::MemorySemanticsMask::SequentiallyConsistent |
                         __spv::MemorySemanticsMask::WorkgroupMemory);
  } else {
    flags = static_cast<uint32_t>(__spv::MemorySemanticsMask::SequentiallyConsistent |
                       __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                       __spv::MemorySemanticsMask::WorkgroupMemory);
  }
  __spirv_ControlBarrier(__spv::Scope::Workgroup, __spv::Scope::Workgroup,
                           flags);
#else
  #warning device barrier called on CPU, this should not happen
#endif
}
}
}
}

#endif
