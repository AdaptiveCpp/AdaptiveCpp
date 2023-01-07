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

#include <set>

namespace hipsycl {
namespace rt {

// Only supports up to 64-256 massive allocations simultaneously, in order to
// balance driver overhead from calling `useResources(_:)` on every encoder.
// Apple GPU virtual addresses start at either 0x11_00000000 or 0x15_00000000
// (https://github.com/AsahiLinux/docs/wiki/HW:AGX)
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

// TODO: Ensure you transfer all the relevant doc comments from the Swift
// prototype.

class metal_heap_pool;
class metal_heap_block;

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

class metal_heap_pool
{
public:

  // TODO: Find a more efficient way to cache the buffer list.
  NS::SharedPtr<NS::Array> extract_buffers();

private:
  friend class metal_allocator;
  
  MTL::Device* _dev;
  uint64_t _cpu_base_va;
  uint64_t _gpu_base_va;
  int64_t _physical_memory_limit;
  int64_t _virtual_memory_limit;
  
  // Borrowed from PyTorch MPSAllocator. We don't cache the buffers; they are
  // immediately returned to the heap. This helps reduce fragmentation, as we
  // can't afford to exceed device memory (unlike PyTorch).
  int64_t kMaxSmallAlloc;
  int64_t kMinLargeAlloc;
  int64_t kSmallHeap;
  int64_t kLargeHeap;
  
  
};

class metal_heap_block
{
public:
  metal_heap_block(MTL::Device* device,
                   uint64_t cpu_base_va,
                   uint64_t gpu_base_va,
                   int64_t size);
  
  // Caller must ensure there's enough space before allocating.
  void* allocate(int64_t size);
  
  // Crashes if the pointer doesn't belong in this heap.
  void deallocate(void* usm_pointer);
  
  // Returns -1 if not found.
  int64_t get_offset(void* usm_pointer);
  
  // For debugging purposes only.
  void validate_sorted();

private:
  friend class metal_heap_pool;

  NS::SharedPtr<MTL::Heap> _heap;
  NS::SharedPtr<MTL::Buffer> _buffer;
  uint64_t _cpu_base_va;
  uint64_t _gpu_base_va;
  uint64_t _heap_va;
  int64_t _available_size;
  std::unordered_map<int64_t, NS::SharedPtr<MTL::Buffer>> _phantom_buffers;
  
  // Heap used size isn't a reliable way to find used size. There's a bug where
  // it returns 4 GB when you have a 4 GB heap, even though no resources are
  // allocated. `maxAvailableSize(alignment:)` still says you can allocate stuff
  // on the heap.
  int64_t _used_size = 0;
  
  struct allocation_t {
    int64_t offset;
    int64_t size;
  };
  struct allocation_compare_t {
    constexpr bool operator()(const allocation_t& lhs,
                              const allocation_t& rhs) const {
      return lhs.offset < rhs.offset;
    }
  };
  std::set<allocation_t, allocation_compare_t> _allocations;
};

}
}

#endif
