/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_MEM_FENCE_HPP
#define HIPSYCL_MEM_FENCE_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

template<access::fence_space, access::mode>
struct mem_fence_impl
{
  HIPSYCL_KERNEL_TARGET
  static void mem_fence()
  {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
    __threadfence();
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
    __spirv_MemoryBarrier(__spv::Scope::Device,
                          __spv::MemorySemanticsMask::SequentiallyConsistent |
                          __spv::MemorySemanticsMask::CrossWorkgroupMemory |
                          __spv::MemorySemanticsMask::WorkgroupMemory);
#else
    // TODO What about CPU?
#endif
  }

};

template<access::mode M>
struct mem_fence_impl<access::fence_space::local_space, M>
{
  HIPSYCL_KERNEL_TARGET
  static void mem_fence()
  {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
    __threadfence_block();
#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
    __spirv_MemoryBarrier(
        __spv::Scope::Workgroup,
        static_cast<uint32_t>(
            __spv::MemorySemanticsMask::SequentiallyConsistent |
            __spv::MemorySemanticsMask::WorkgroupMemory));
#endif
  }
};



template <
  access::fence_space Fence_space = access::fence_space::global_and_local,
  access::mode Mode = access::mode::read_write
>
HIPSYCL_KERNEL_TARGET
inline void mem_fence()
{
  static_assert(Mode == access::mode::read ||
                Mode == access::mode::write ||
                Mode == access::mode::read_write,
                "mem_fence() is only allowed for read, write "
                "or read_write access modes.");
  mem_fence_impl<Fence_space, Mode>::mem_fence();
}

template<access::mode Mode>
HIPSYCL_KERNEL_TARGET
inline void mem_fence(access::fence_space space)
{
  if(space == access::fence_space::local_space)
    mem_fence<access::fence_space::local_space, Mode>();

  else if(space == access::fence_space::global_space)
    mem_fence<access::fence_space::global_space, Mode>();

  else if(space == access::fence_space::global_and_local)
    mem_fence<access::fence_space::global_and_local, Mode>();
}

}
}
}

#endif
