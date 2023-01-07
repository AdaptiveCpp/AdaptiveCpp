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
 
#include <cassert>
#include <cstdint>
#include <string>
#include <limits>
#include <iterator>
#include <bit>

#include <Metal/Metal.hpp>
#include <sys/sysctl.h>

#include "hipSYCL/runtime/metal/metal_allocator.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

int64_t fetch_sysctl_property(const char *name);

// Rounds an integer up to the nearest power of 2.
int64_t round_up_to_power_of_2(int64_t input);

// Rounds an integer down to the nearest power of 2.
int64_t round_down_to_power_of_2(int64_t input);

metal_allocator::metal_allocator(const metal_hardware_context *device)
  : _dev{device->get_metal_device()}
{
  
}

// TODO: Change all `1 << 14` and `16384` to `vm_page_size`.
metal_heap_block::metal_heap_block(MTL::Device* device,
                                   uint64_t cpu_base_va,
                                   uint64_t gpu_base_va,
                                   int64_t size) {
  auto heapDesc = NS::TransferPtr(MTL::HeapDescriptor::alloc()->init());
  heapDesc->setHazardTrackingMode(MTL::HazardTrackingModeUntracked);
  heapDesc->setStorageMode(MTL::StorageModeShared);
  heapDesc->setType(MTL::HeapTypeAutomatic);
  heapDesc->setSize(0);
  auto heap = device->newHeap(heapDesc.get());
  if(!heap) {
    throw error_type::memory_allocation_error;
  }
  this->_heap = NS::TransferPtr(heap);
  heap->setPurgeableState(MTL::PurgeableStateEmpty);
  
  // This should never take more than 2, possibly 3, tries.
  auto target_address = gpu_base_va;
  MTL::Buffer* final_buffer;
  for (int i = 0; i < 1000; ++i) {
    auto delta = target_address - gpu_base_va;
    assert(delta >= 0 && "Tried to allocate below GPU base VA.");
    
    auto usm_pointer = (void*)(cpu_base_va + delta);
    auto buffer = device->newBuffer(/*bytesNoCopy*/usm_pointer, size, 0, NULL);
    if (!buffer) {
      buffer->release();
      throw error_type::memory_allocation_error;
    }
    
    if (buffer->gpuAddress() != target_address) {
      target_address = buffer->gpuAddress();
      continue;
    }
    final_buffer = buffer;
    break;
  }
  assert(final_buffer && "Took too many attempts to allocate USM heap.");
  
  auto gpu_address = final_buffer->gpuAddress();
  this->_buffer = NS::TransferPtr(final_buffer);
  this->_gpu_base_va = gpu_address;
  this->_cpu_base_va = cpu_base_va + (gpu_address - gpu_base_va);
  this->_available_size = heap->maxAvailableSize(1 << 14);
  
  // The heap doesn't give us its base address, but we can find a way to
  // acquire it anyway. First allocate a buffer, and see how it impacts
  // available size. The second buffer should come right after it, and max
  // available size should decrease by the same amount.
  //
  // If we ever see different behavior, we still have enough information to
  // reconstruct the base VA.
  assert((_available_size == heap->size()) && "Heap available size mismatch.");
  assert((_available_size == size) && "Heap available size mismatch.");
  assert((heap->currentAllocatedSize() == size) && "Heap available size mismatch.");
  
  // TODO: Make all the assertions into proper errors, which compile during
  // release builds.
  {
    // TODO: We can take a much more deterministic and simple approach. Make a
    // buffer that consumes the entire heap's memory (`_available_size`). Take
    // its `gpuAddress()`, then deallocate it.
    
    // Check that available size decreased by 16384.
    auto buffer1 = NS::TransferPtr(heap->newBuffer(1 << 14, 0));
    assert((buffer1->heapOffset() == 0) && "Unexpected heap allocation behavior.");
    int64_t new_available_size1 = heap->maxAvailableSize(1 << 14);
    int64_t expected_size1 = this->_available_size - (1 << 14);
    assert((new_available_size1 == expected_size1) && "Unexpected heap allocation behavior.");
    
    // Check that available size decreased by 16384.
    auto buffer2 = NS::TransferPtr(heap->newBuffer(1 << 14, 0));
    assert((buffer2->heapOffset() == 0) && "Unexpected heap allocation behavior.");
    int64_t new_available_size2 = heap->maxAvailableSize(1 << 14);
    int64_t expected_size2 = this->_available_size - 2 * (1 << 14);
    assert((new_available_size2 == expected_size2) && "Unexpected heap allocation behavior.");
    
    // Check that size decreased because buffers were allocated from the start,
    // not end, of the heap.
    uint64_t address1 = buffer1->gpuAddress();
    uint64_t address2 = buffer2->gpuAddress();
    assert((address1 + 16384 == address2) && "Unexpected heap allocation behavior.");
    this->_heap_va = address1;
  }
  
  // Check that the buffers deallocated.
  auto new_available_size = heap->maxAvailableSize(1 << 14);
  assert((this->_available_size == new_available_size) && "Unexpected heap allocation behavior.");
}

void* metal_heap_block::allocate(int64_t size) {
  assert(size <= _available_size && "Didn't check size before allocating buffer from heap.");
  
  // Use the buffer's allocated size, never its actual length, when doing size
  // calculations.
  MTL::Buffer* phantom_buffer = _heap->newBuffer(size, 0);
  int64_t offset = phantom_buffer->gpuAddress() - _heap_va;
  int64_t phantom_size = phantom_buffer->allocatedSize();
  _phantom_buffers.insert({ offset, NS::TransferPtr(phantom_buffer) });
  
  allocation_t alloc{ offset, phantom_size };
  _allocations.insert(alloc);
  validate_sorted();
  
  _used_size += phantom_size;
  _available_size = _heap->maxAvailableSize(1 << 14);
  return (void*)(_cpu_base_va + offset);
}

void metal_heap_block::deallocate(void* usm_pointer) {
  int64_t offset = (uint64_t)usm_pointer - _cpu_base_va;
  auto phantom_it = _phantom_buffers.find(offset);
  assert((phantom_it != _phantom_buffers.end()) && "Pointer did not originate from this heap.");
  MTL::Buffer* phantom_buffer = (*phantom_it).second.get();
  int64_t phantom_size = phantom_buffer->allocatedSize();
  _phantom_buffers.erase(phantom_it);
  
  allocation_t expected_alloc{ offset, phantom_size };
  auto alloc_it = _allocations.lower_bound(expected_alloc);
  assert((alloc_it != _allocations.end()) && "Allocation not found in sorted list.");
  assert((*alloc_it == expected_alloc) && "Allocation not found in sorted list.");
  _allocations.erase(alloc_it);
  
  _used_size -= phantom_size;
  _available_size = _heap->maxAvailableSize(1 << 14);
}

// TODO: Copy doc comment from Swift prototype that explains why we want to
// batch pointer searches.
//
// TODO: Make a more efficient traversal method after debugging, which is
// batchable but returns failures. Unless the compiler will always inline
// this. You need to profile and see how much overhead it invokes.
//
// Defer the actual optimization until later. In fact, it can be a follow-up
// pull request. Just quantify the performance impact.
int64_t metal_heap_block::get_offset(void* usm_pointer) {
  int64_t offset = (uint64_t)usm_pointer - _cpu_base_va;
  allocation_t dummy_alloc{ offset, 0 };
  auto upper_bound_it = _allocations.upper_bound(dummy_alloc);
  if (upper_bound_it == _allocations.begin()) {
    return -1;
  }
  allocation_t alloc = *(prev(upper_bound_it));
  if (alloc.offset <= offset &&
      offset < alloc.offset + alloc.size) {
    return offset;
  } else {
    return -1;
  }
}

void metal_heap_block::validate_sorted() {
  if (_allocations.size() > 1) {
    auto start = _allocations.begin();
    auto end = prev(_allocations.end());
    for (auto it = start; it != end; ++it) {
      auto element1 = *it;
      auto element2 = *(next(it));
      assert((element1.offset + element1.size <= element2.offset) && "Allocations are not sorted.");
    }
  }
}

int64_t fetch_sysctl_property(const char *name) {
  int64_t ret = 0;
  size_t size = sizeof(int64_t);
  int error = sysctlbyname(name, &ret, &size, NULL, 0);
  assert((error == 0) && "sysctl failed.");
  return ret;
}

int64_t round_up_to_power_of_2(int64_t input) {
  uint64_t to_count = (0 > (input - 1)) ? 0 : (input - 1);
  return 1 << (64 - std::__countl_zero(to_count));
}

int64_t round_down_to_power_of_2(int64_t input) {
  uint64_t to_count = input;
  return 1 << (64 - 1 - std::__countl_zero(to_count));
}

}
}
