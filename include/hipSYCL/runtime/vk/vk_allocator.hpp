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

#pragma once

#include "../allocator.hpp"
#include <mutex>
#include <unordered_map>
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_hardware_context;

struct vk_alloc_info {
  vk_alloc_info() = delete;

  vk::DeviceAddress _base_ptr;
  size_t bytes;
  vk::raii::Buffer _buffer;
  vk::raii::DeviceMemory _dev_mem;
};

class vk_allocator : public backend_allocator {
public:
  vk_allocator() {}
  vk_allocator(vk_hardware_context *hw_ctx, rt::device_id dev);

  void *raw_allocate(size_t min_alignment, size_t size_bytes,
                     const allocation_hints &hints = {}) override;

  void *
  raw_allocate_optimized_host(size_t min_alignment, size_t bytes,
                              const allocation_hints &hints = {}) override;

  void raw_free(void *mem) override;

  void *raw_allocate_usm(size_t bytes,
                         const allocation_hints &hints = {}) override;
  bool is_usm_accessible_from(backend_descriptor b) const override;

  result query_pointer(const void *ptr, pointer_info &out) const override;

  result mem_advise(const void *addr, std::size_t num_bytes,
                    int advise) const override;

  device_id get_device() const override;

  std::size_t get_global_mem_size() const;

  vk_alloc_info *find_alloc_info(vk::DeviceAddress ptr);

  // Creates a single buffer backed by allocated device memory, and created with
  // properties which allow its device address returned
  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
  create_device_address_buffer(vk::DeviceSize size);

  // Given a list of uniform buffers sets scraped from SPIR-V reflection
  // allocates enough memory for all of then, then creates buffers into the
  // memory which are bound the appropriate offset
  std::tuple<std::vector<vk::raii::Buffer>, std::vector<vk::DeviceSize>,
             vk::raii::DeviceMemory>
  create_uniform_buffers(std::vector<vk::DeviceSize> sizes);

private:
  uint32_t find_memory_type(vk::MemoryPropertyFlags properties,
                            uint32_t type_filter = UINT32_MAX) const;

  rt::device_id _dev;
  vk_hardware_context *_hw_ctx; // non owning
  mutable std::mutex _mutex;

  vk::PhysicalDeviceMemoryProperties _mem_properties;
  std::unordered_map<vk::DeviceAddress, vk_alloc_info> _allocs;
};

} // namespace rt
} // namespace hipsycl
