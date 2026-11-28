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

// This header describes the native objects that the Vulkan backend hands out
// for interoperability. It is included by user code through the SYCL interop
// API, and therefore deliberately only depends on the Vulkan C headers instead
// of the vulkan-hpp/vulkan-raii C++ bindings that the backend implementation
// uses internally.
//
// To avoid leaking vk backend implementation details to other parts of the
// codebase that could include this file, the interfaces below are declared
// which the runtime classes of the Vulkan backend implement.

#include <cstdint>

#include <vulkan/vulkan_core.h>

namespace hipsycl {
namespace rt {

// Native objects describing a SYCL device of the Vulkan backend.
struct vk_device_native_handle {
  // Instance the backend has created, corresponds to sycl::platform
  VkInstance instance;
  // Physical device, corresponds to sycl::device
  VkPhysicalDevice physical_device;
  // Logical device the backend has created for this physical device,
  // corresponds to sycl::context
  VkDevice device;
  // Index of the queue family that the backend submits work to
  uint32_t queue_family_index;
};

// Describes a point in the execution of a Vulkan queue using a timeline
// semaphore and the value it attains once the work has completed.
struct vk_event_native_handle {
  VkSemaphore semaphore;
  uint64_t value;
  // Logical device the semaphore belongs to. sycl::make_event() uses it to find
  // the device of the context the semaphore belongs to, so it may only be left
  // as VK_NULL_HANDLE if the context contains a single Vulkan device.
  VkDevice device = VK_NULL_HANDLE;
};

// Describes the Vulkan buffer backing a USM allocation. Device USM pointers of
// the Vulkan backend are device addresses obtained from
// VK_KHR_buffer_device_address, and as such always refer to a region of a
// VkBuffer.
struct vk_allocation_native_handle {
  VkBuffer buffer;
  // Offset of the queried pointer inside the buffer, in bytes
  VkDeviceSize offset;
  // Number of bytes between the queried pointer and the end of the allocation
  VkDeviceSize size;
};

// Implemented by vk_hardware_context
class vk_native_device_provider {
public:
  virtual vk_device_native_handle get_native_device_handle() const = 0;
  virtual ~vk_native_device_provider() {}
};

// Implemented by vk_allocator
class vk_native_allocation_provider {
public:
  // Returns false if ptr does not belong to an allocation of this allocator.
  virtual bool
  get_native_allocation(const void *ptr,
                        vk_allocation_native_handle &out) const = 0;
  virtual ~vk_native_allocation_provider() {}
};

// Implemented by vk_node_event and vk_imported_node_event
class vk_native_event_provider {
public:
  virtual vk_event_native_handle get_native_event_handle() const = 0;
  virtual ~vk_native_event_provider() {}
};

} // namespace rt
} // namespace hipsycl
