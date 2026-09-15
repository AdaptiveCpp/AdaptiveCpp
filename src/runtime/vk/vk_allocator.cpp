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

std::pair<uint32_t, vk::MemoryPropertyFlags>
vk_allocator::find_memory_type(vk::MemoryPropertyFlags properties,
                               uint32_t type_filter) const {
  for (uint32_t i = 0; i < _mem_properties.memoryTypeCount; i++) {
    if ((type_filter & (1 << i)) &&
        (_mem_properties.memoryTypes[i].propertyFlags & properties) ==
            properties) {
      return std::make_pair(i, _mem_properties.memoryTypes[i].propertyFlags);
    }
  }

  return std::make_pair(UINT32_MAX, vk::MemoryPropertyFlags{});
}

std::size_t vk_allocator::get_global_mem_size() const {
  // Return the size of the heap device USM pointers will be allocated from
  constexpr vk::MemoryPropertyFlags preferred_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent |
      vk::MemoryPropertyFlagBits::eDeviceLocal;

  auto [type_index, _] = find_memory_type(preferred_mem_prop_flags);
  if (type_index != UINT32_MAX) {
    uint32_t heap_index = _mem_properties.memoryTypes[type_index].heapIndex;
    return _mem_properties.memoryHeaps[heap_index].size;
  }

  constexpr vk::MemoryPropertyFlags required_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;
  std::tie(type_index, std::ignore) = find_memory_type(required_mem_prop_flags);
  uint32_t heap_index = _mem_properties.memoryTypes[type_index].heapIndex;
  return _mem_properties.memoryHeaps[heap_index].size;
}

std::tuple<std::vector<vk::raii::Buffer>, std::vector<vk::DeviceSize>,
           vk::raii::DeviceMemory>
vk_allocator::create_uniform_buffers(std::vector<vk::DeviceSize> sizes) {
  const auto &device = _hw_ctx->get_device();
  std::vector<vk::raii::Buffer> buffers;
  std::vector<vk::DeviceSize> offsets;

  vk::DeviceSize total_size = 0;
  vk::DeviceSize max_alignment = 0;
  uint32_t type_bitmask = UINT32_MAX;

  // First create the buffers and extract requirements for the memory allocation
  const vk::DeviceSize min_uniform_align =
      _hw_ctx->get_min_uniform_buffer_offset_alignment();
  for (size_t i = 0; i < sizes.size(); i++) {
    vk::BufferCreateInfo buffer_info{{},
                                     sizes[i],
                                     vk::BufferUsageFlagBits::eUniformBuffer,
                                     vk::SharingMode::eExclusive};
    vk::raii::Buffer buffer(device, buffer_info);
    vk::MemoryRequirements mem_reqs = buffer.getMemoryRequirements();

    // Find the alignment of this buffer, and add padding
    vk::DeviceSize align = std::max(mem_reqs.alignment, min_uniform_align);
    vk::DeviceSize padding = (align - (total_size % align)) % align;
    total_size += padding;
    offsets.push_back(total_size);

    total_size += mem_reqs.size;
    type_bitmask = type_bitmask & mem_reqs.memoryTypeBits;

    buffers.push_back(std::move(buffer));
  }

  constexpr vk::BufferUsageFlags usage_flags =
      vk::BufferUsageFlagBits::eUniformBuffer;

  constexpr vk::MemoryPropertyFlags required_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent;

  constexpr vk::MemoryPropertyFlags preferred_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent |
      vk::MemoryPropertyFlagBits::eDeviceLocal;

  // Try use preferred memory properties, and if not available fall back to
  // required memory properties.
  auto [mem_type_index, _] =
      find_memory_type(preferred_mem_prop_flags, type_bitmask);
  if (mem_type_index == UINT32_MAX) {
    std::tie(mem_type_index, std::ignore) =
        find_memory_type(required_mem_prop_flags, type_bitmask);
    if (mem_type_index == UINT32_MAX) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::memory_allocation),
          "Could not allocate memory"};
    }
  }

  vk::MemoryAllocateInfo alloc_info{total_size, mem_type_index};
  vk::raii::DeviceMemory buffer_mem(device, alloc_info);
  for (size_t i = 0; i < buffers.size(); i++) {
    buffers[i].bindMemory(buffer_mem, offsets[i]);
  }

  HIPSYCL_DEBUG_INFO << "vk_allocator: created " << buffers.size()
                     << " uniform buffers backed by a " << total_size
                     << " byte memory allocation" << std::endl;

  return {std::move(buffers), std::move(offsets), std::move(buffer_mem)};
}

std::tuple<vk::raii::Buffer, vk::raii::DeviceMemory, vk::MemoryPropertyFlags>
vk_allocator::create_device_address_buffer(vk::DeviceSize size) {
  constexpr vk::BufferUsageFlags usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc |
      vk::BufferUsageFlagBits::eTransferDst |
      vk::BufferUsageFlagBits::eShaderDeviceAddress;

  // This function allocates a device USM pointer, so make an effort to
  // use a device pointer. Ideally this will also be from host visible and
  // coherent memory so we can map it.
  constexpr vk::MemoryPropertyFlags required_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eDeviceLocal;

  constexpr vk::MemoryPropertyFlags preferred_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent |
      vk::MemoryPropertyFlagBits::eDeviceLocal;

  vk::BufferCreateInfo buffer_info{
      {}, size, usage_flags, vk::SharingMode::eExclusive};
  const auto &device = _hw_ctx->get_device();
  vk::raii::Buffer buffer(device, buffer_info);

  // Try use preferred memory properties, and if not available fall back to
  // required memory properties.
  vk::MemoryRequirements mem_reqs = buffer.getMemoryRequirements();
  auto [mem_type_index, mem_flags] =
      find_memory_type(preferred_mem_prop_flags, mem_reqs.memoryTypeBits);
  if (mem_type_index == UINT32_MAX) {
    std::tie(mem_type_index, mem_flags) =
        find_memory_type(required_mem_prop_flags, mem_reqs.memoryTypeBits);
    if (mem_type_index == UINT32_MAX) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::memory_allocation),
          "Could not allocate memory"};
    }
  }

  vk::MemoryAllocateInfo alloc_info{mem_reqs.size, mem_type_index};

  // vkHpp doesn't seem to like this, so pointer chain manually
  VkMemoryAllocateFlagsInfo flags_info{};
  flags_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  flags_info.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  alloc_info.pNext = &flags_info;

  vk::raii::DeviceMemory buffer_mem(device, alloc_info);
  buffer.bindMemory(buffer_mem, 0);

  return std::make_tuple(std::move(buffer), std::move(buffer_mem), mem_flags);
}

vk_alloc_info *vk_allocator::find_user_alloc(vk::DeviceAddress ptr) {
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

vk_alloc_info *vk_allocator::staging_allocate(size_t size_bytes) {
  constexpr vk::BufferUsageFlags usage_flags =
      vk::BufferUsageFlagBits::eTransferSrc |
      vk::BufferUsageFlagBits::eTransferDst;

  // Prefer host cached if possible for fastest access from host, as
  // we will not access the staging buffer on device
  constexpr vk::MemoryPropertyFlags required_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent;

  constexpr vk::MemoryPropertyFlags preferred_mem_prop_flags =
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent |
      vk::MemoryPropertyFlagBits::eHostCached;

  const auto &device = _hw_ctx->get_device();
  vk::BufferCreateInfo buffer_info{
      {}, size_bytes, usage_flags, vk::SharingMode::eExclusive};
  vk::raii::Buffer buffer(device, buffer_info);

  // Try use preferred memory properties, and if not available fall back to
  // required memory properties.
  vk::MemoryRequirements mem_reqs = buffer.getMemoryRequirements();
  auto [mem_type_index, mem_flags] =
      find_memory_type(preferred_mem_prop_flags, mem_reqs.memoryTypeBits);
  if (mem_type_index == UINT32_MAX) {
    std::tie(mem_type_index, mem_flags) =
        find_memory_type(required_mem_prop_flags, mem_reqs.memoryTypeBits);
    if (mem_type_index == UINT32_MAX) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::memory_allocation),
          "Could not allocate memory"};
    }
  }

  vk::MemoryAllocateInfo alloc_info{mem_reqs.size, mem_type_index};
  vk::raii::DeviceMemory buffer_mem(device, alloc_info);
  buffer.bindMemory(buffer_mem, 0);

  HIPSYCL_DEBUG_INFO << "vk_allocator: staging allocated " << size_bytes
                     << " bytes at " << std::hex << *buffer << std::dec
                     << std::endl;

  return new vk_alloc_info{
      vk_alloc_type::STAGING, 0,        size_bytes, std::move(buffer),
      std::move(buffer_mem),  mem_flags};
}

void *vk_allocator::raw_allocate(size_t, size_t size_bytes,
                                 const allocation_hints &) {
  std::lock_guard<std::mutex> lock{_mutex};

  auto [buffer, device_mem, mem_flags] =
      create_device_address_buffer(size_bytes);

  vk::BufferDeviceAddressInfo addr_info{buffer};
  vk::DeviceAddress ptr = _hw_ctx->get_device().getBufferAddress(addr_info);

  vk_alloc_info alloc_info{
      vk_alloc_type::USER,   ptr,      size_bytes, std::move(buffer),
      std::move(device_mem), mem_flags};
  _allocs.insert({ptr, std::move(alloc_info)});

  HIPSYCL_DEBUG_INFO << "vk_allocator: user allocated " << size_bytes
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

  HIPSYCL_DEBUG_INFO << "vk_allocator: freed " << std::hex << mem << std::dec
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
