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

#include "hipSYCL/runtime/metal/metal_allocator.hpp"

#include <Metal/Metal.hpp>

namespace hipsycl {
namespace rt {

metal_allocator::metal_allocator(MTL::Device* device, const device_id &id)
  : _device{device}, _device_id{id}
{}

metal_allocator::~metal_allocator() = default;

void* metal_allocator::raw_allocate(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto storage_mode = MTL::ResourceStorageModePrivate;
  auto buffer = _device->newBuffer(size_bytes, storage_mode);
  void* gpu_ptr = reinterpret_cast<void*>(buffer->gpuAddress());
  auto block = usm_block{
    .buffer = buffer,
    .size = size_bytes,
    .alloc_type = usm_alloc_type::device
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[gpu_ptr] = block;
  return gpu_ptr;
}

void *metal_allocator::raw_allocate_usm(
  size_t size_bytes,
  const allocation_hints &hints)
{
  auto storage_mode = MTL::ResourceStorageModeShared;
  auto buffer = _device->newBuffer(size_bytes, storage_mode);
  void* host_ptr = buffer->contents();
  auto block = usm_block{
    .buffer = buffer,
    .size = size_bytes,
    .alloc_type = usm_alloc_type::shared
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[host_ptr] = block;
  return host_ptr;
}

void *
metal_allocator::raw_allocate_optimized_host(
  size_t min_alignment, size_t size_bytes,
  const allocation_hints &hints)
{
  auto storage_mode = MTL::ResourceStorageModeShared;
  auto buffer = _device->newBuffer(size_bytes, storage_mode);
  void* host_ptr = buffer->contents();
  auto block = usm_block{
    .buffer = buffer,
    .size = size_bytes,
    .alloc_type = usm_alloc_type::host
  };
  std::lock_guard<std::mutex> lock{_mutex};
  _ptr_to_block[host_ptr] = block;
  return host_ptr;
}

void metal_allocator::raw_free(void *mem)
{
  if (!mem) return;

  std::lock_guard<std::mutex> lock{_mutex};
  auto it = _ptr_to_block.find(mem);
  if (it != _ptr_to_block.end()) {
    if(it->second.buffer) {
      it->second.buffer->release();
    } else {
      std::free(mem);
    }
    _ptr_to_block.erase(it);
  }
}

bool metal_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  return b.id == backend_id::metal;
}

result metal_allocator::query_pointer(
  const void *ptr,
  pointer_info &out) const
{
  memset(&out, 0, sizeof(pointer_info));
  out.dev = _device_id;
  if (!ptr) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Null pointer queried"});
  }
  auto [buffer, offset, alloc_type] = get_usm_block(ptr);
  if (alloc_type == usm_alloc_type::undefined) {
    return make_error(__acpp_here(),
      error_info{"metal_allocator: Pointer is unknown"});
  }
  if (alloc_type == usm_alloc_type::host) {
    out.is_optimized_host = true;
    return make_success();
  }
  if (alloc_type == usm_alloc_type::shared) {
    out.is_usm = true;
    return make_success();
  }

  return make_success();
}

result metal_allocator::mem_advise(
  const void *addr, std::size_t num_bytes,
  int advise) const
{
  return make_success();
}

device_id metal_allocator::get_device() const {
  return _device_id;
}

std::tuple<MTL::Buffer*, size_t, metal_allocator::usm_alloc_type> metal_allocator::get_usm_block(const void* ptr) const {
  std::lock_guard<std::mutex> lock{_mutex};
  if (_ptr_to_block.empty()) {
    return {nullptr, 0, usm_alloc_type::undefined};
  }
  auto it = _ptr_to_block.upper_bound(const_cast<void*>(ptr));
  if (it == _ptr_to_block.begin()) {
    return {nullptr, 0, usm_alloc_type::undefined};
  }
  --it;
  const usm_block& block = it->second;
  size_t offset = static_cast<const char*>(ptr) -
          static_cast<const char*>(it->first);
  if (offset < block.size) {
    return {block.buffer, offset, block.alloc_type};
  }
  return {nullptr, 0, usm_alloc_type::undefined};
}

} // namespace rt
} // namespace hipsycl
