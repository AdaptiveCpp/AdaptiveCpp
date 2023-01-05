/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_METAL_ALLOCATOR_HPP
#define HIPSYCL_METAL_ALLOCATOR_HPP

#include "../allocator.hpp"
#include "metal_hardware_manager.hpp"

#include <Metal/Metal.hpp>

namespace hipsycl {
namespace rt {

// Only supports up to 128 massive allocations simultaneously, in order to
// balance driver overhead from calling `useResources(_:)` on every encoder.
// Apple GPU virtual addresses start at either 0x11_00000000 or 0x15_00000000
// (https://github.com/AsahiLinux/docs/wiki/HW:AGX)
//
// Each chunk should keep a retain count. That way, when the GPU isn't using it,
// we can avoid `useResource`ing it. Also, make a dictionary of live chunks and
// query whether one's been created there. Metal's virtual address assignment
// tells use where to place new allocations; we don't need a custom `malloc`
// implementation.
//
// macOS:
// Allocate 1 TB of virtual memory, enough to fill 40 bits of GPU VA. Or,
// allocate 3x the device's RAM size, whichever is larger. This is nowhere near
// the macOS virtual address space (128 TB), so many apps can use hipSYCL
// simultaneously. Since we have more breathing room than iOS, USM addresses
// start at 0x0 to ensure forward compatibility. The 3x RAM option also ensures
// forward compatibility.
// 0x00_00000000..<0xFF_FFFFFFFF
//
// iOS:
// Try to allocate as much VM space as possible. First go by 1/2 the space from
// the number of VM bits, re-guessing at lower powers of 2. Recursively
// subdivide the remainder into 16 parts. Finally, take 2/3. This eats up ~10%
// of the iOS virtual address space, so probably only one app can use hipSYCL
// simultaneously.
// 0x11_00000000..<0x1E_00000000
//
//  let globalMemorySize = fetchSysctlProperty(name: "hw.memsize")
//  let virtualAddressBits = fetchSysctlProperty(
//    name: "machdep.virtual_address_size")
//  let globalMemory = (65 * (globalMemorySize >> 20) / 100) << 20
//
//  // Conservative limit of 6% maximum allocatable memory.
//  let virtualMemory = (1 << virtualAddressBits) / 16
class metal_allocator : public backend_allocator
{
public:
  metal_allocator(const metal_hardware_context* dev);

//  virtual void* allocate(size_t min_alignment, size_t size_bytes) override;
//
//  virtual void *allocate_optimized_host(size_t min_alignment,
//                                        size_t bytes) override;
//
//  virtual void free(void *mem) override;
//
//  virtual void *allocate_usm(size_t bytes) override;
//  virtual bool is_usm_accessible_from(backend_descriptor b) const override;
//
//  virtual result query_pointer(const void* ptr, pointer_info& out) const override;
//
//  virtual result mem_advise(const void *addr, std::size_t num_bytes,
//                            int advise) const override;
private:
  MTL::Device* _dev;
  int64_t _max_allocated_size;
  int64_t _vm_region_size;
  int64_t _cpu_base_va;
  int64_t _gpu_base_va;
  

};

}
}

#endif
