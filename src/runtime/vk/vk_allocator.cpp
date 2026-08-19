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
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/sycl/exception.hpp"

#include "hipSYCL/runtime/vk/vk_allocator.hpp"
#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"
#include <cstddef>

namespace hipsycl {
namespace rt {
vk_allocator::vk_allocator(vk_hardware_context *hw_ctx, rt::device_id dev)
    : _dev{dev}, _hw_ctx(hw_ctx) {
  _mem_properties = _hw_ctx->get_physical_device().getMemoryProperties();
}

uint32_t vk_allocator::find_memory_type(vk::MemoryPropertyFlags properties,
                                        uint32_t type_filter) const {
  for (uint32_t i = 0; i < _mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (_mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return i;
    }
  }

  print_error(__acpp_here(),
              error_info{"failed to find suitable memory type!"});
  return UINT32_MAX;
}

std::size_t vk_allocator::get_global_mem_size() const {
  constexpr vk::MemoryPropertyFlags mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent;
  uint32_t type_index = find_memory_type(mem_prop_flags);
  uint32_t heap_index = _mem_properties.memoryTypes[type_index].heapIndex;
  return _mem_properties.memoryHeaps[heap_index].size;
}

std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
vk_allocator::create_buffer(vk::DeviceSize size,
                            vk::BufferUsageFlags usage_flags) {
  constexpr vk::MemoryPropertyFlags mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent;

  vk::BufferCreateInfo buffer_info{
      {}, size, usage_flags, vk::SharingMode::eExclusive};
  const auto &device = _hw_ctx->get_device();
  vk::raii::Buffer buffer(device, buffer_info);

  vk::MemoryRequirements mem_reqs = buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo alloc_info{
      mem_reqs.size, find_memory_type(mem_prop_flags, mem_reqs.memoryTypeBits)};

  // vkHpp doesn't seem to like this, so pointer chain manually
  VkMemoryAllocateFlagsInfo flags_info{};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  alloc_info.pNext = &flags_info;

  vk::raii::DeviceMemory buffer_mem(device, alloc_info);
  buffer.bindMemory(buffer_mem, 0);

  return std::make_pair(std::move(buffer), std::move(buffer_mem));
}

vk_alloc_info *vk_allocator::find_alloc_info(vk::DeviceAddress ptr) {
  std::lock_guard<std::mutex> lock{_mutex};
  // Try to find quickly from base used as key to map
  if (_allocs.count(ptr)) {
    vk_alloc_info &alloc_info = _allocs.find(ptr)->second;
    return &alloc_info;
  }

  // Try to find ptr anywhere in range of allocate addresses
  for (auto &alloc : _allocs) {
    vk_alloc_info &alloc_info = alloc.second;
    vk::DeviceAddress base = alloc_info._base_ptr;
    vk::DeviceAddress end = base + alloc_info.bytes;
    if (ptr > base && ptr < end) {
      return &alloc_info;
    }
  }

  return nullptr;
}

void *vk_allocator::raw_allocate(size_t, size_t size_bytes,
                                 const allocation_hints &) {
  std::lock_guard<std::mutex> lock{_mutex};

  constexpr vk::BufferUsageFlags usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc |
      vk::BufferUsageFlagBits::eTransferDst |
      vk::BufferUsageFlagBits::eShaderDeviceAddress;
  auto [buffer, device_mem] = create_buffer(size_bytes, usage_flags);

  vk::BufferDeviceAddressInfo addr_info{buffer};
  vk::DeviceAddress ptr = _hw_ctx->get_device().getBufferAddress(addr_info);

  vk_alloc_info alloc_info{ptr, size_bytes, std::move(buffer),
                           std::move(device_mem)};
  _allocs.insert({ptr, std::move(alloc_info)});

  HIPSYCL_DEBUG_INFO << "vk_allocator: allocated " << size_bytes
                     << " bytes at 0x" << std::hex << ptr << std::dec
                     << std::endl;
  return reinterpret_cast<void *>(ptr);
}

void *vk_allocator::raw_allocate_optimized_host(size_t, size_t,
                                                const allocation_hints &) {
  // Don't support host USM as virtual pointer from vkMapMemory can't be used
  // inside a kernel as a physical addressing ptr
  throw sycl::exception{
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Vulkan device does not support host USM"};
  return nullptr;
}

void vk_allocator::raw_free(void *mem) {
  std::lock_guard<std::mutex> lock{_mutex};
  auto dev_ptr = reinterpret_cast<vk::DeviceAddress>(mem);
  assert(_allocs.count(dev_ptr));

  HIPSYCL_DEBUG_INFO << "vk_allocator: freed 0x" << std::hex << mem << std::dec
                     << std::endl;
  _allocs.erase(dev_ptr);
}

void *vk_allocator::raw_allocate_usm(size_t, const allocation_hints &) {
  // Don't support shared USM as virtual pointer from vkMapMemory can't be used
  // inside a kernel as a physical addressing ptr
  throw sycl::exception{
      sycl::make_error_code(sycl::errc::feature_not_supported),
      "Vulkan device does not support shared USM"};
  return nullptr;
}

bool vk_allocator::is_usm_accessible_from(backend_descriptor b) const {
  return false;
}

device_id vk_allocator::get_device() const { return _dev; }

result vk_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  std::lock_guard<std::mutex> lock{_mutex};
  out.is_optimized_host = false;
  out.is_from_host_backend = false;
  out.is_usm = false;
  out.native_handle = nullptr;
  out.native_offset = 0;

  auto dev_ptr = reinterpret_cast<vk::DeviceAddress>(ptr);
  if (_allocs.count(dev_ptr)) {
    out.dev = _dev;
    return make_success();
  }

  // Slower path if pointer is at an offset
  for (const auto &alloc : _allocs) {
    const vk_alloc_info &alloc_info = alloc.second;
    vk::DeviceAddress base = alloc_info._base_ptr;
    vk::DeviceAddress end = base + alloc_info.bytes;
    if (dev_ptr >= base && dev_ptr < end) {
      out.dev = _dev;
      return make_success();
    }
  }

  return make_error(
      __acpp_here(),
      error_info{"vk_allocator: Could not find pointer allocation"});
}

result vk_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                int advise) const {
  HIPSYCL_DEBUG_WARNING << "vk_allocator: Ignoring mem_advise() hint"
                        << std::endl;
  return make_success();
}

} // namespace rt
} // namespace hipsycl
