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

#include <Metal/Metal.hpp>
#include <sys/sysctl.h>

#include "hipSYCL/runtime/metal/metal_allocator.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

metal_allocator::metal_allocator(const metal_hardware_context *device)
  : _dev{device->get_metal_device()}
{
  
}

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
  assert(_available_size == heap->size() && "Heap available size mismatch.");
  assert(_available_size == size && "Heap available size mismatch.");
  assert(heap->currentAllocatedSize() == size && "Heap available size mismatch.");
  
  
}

}
}
