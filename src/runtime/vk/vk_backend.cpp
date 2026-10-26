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

#include "hipSYCL/runtime/vk/vk_backend.hpp"
#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include "hipSYCL/runtime/vk/vk_allocator.hpp"
#include "hipSYCL/runtime/vk/vk_event.hpp"
#include "hipSYCL/runtime/vk/vk_native_objects.hpp"
#include "hipSYCL/runtime/vk/vk_queue.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::vk_backend();
}

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() { return "vk"; }

namespace hipsycl {
namespace rt {

namespace {
std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(vk_backend *b, vk_hardware_manager *mgr) {
  return std::make_unique<multi_queue_executor>(*b, [mgr](device_id dev) {
    return std::make_unique<vk_queue>(mgr,
                                      static_cast<std::size_t>(dev.get_id()));
  });
}
} // namespace

vk_backend::vk_backend()
    : _executor([this]() {
        return create_multi_queue_executor(this, &_hw_manager);
      }) {}

api_platform vk_backend::get_api_platform() const { return api_platform::vk; }

hardware_platform vk_backend::get_hardware_platform() const {
  return hardware_platform::vk;
}

backend_id vk_backend::get_unique_backend_id() const { return backend_id::vk; }

backend_hardware_manager *vk_backend::get_hardware_manager() const {
  return &_hw_manager;
}

backend_executor *vk_backend::get_executor(device_id) const {
  return _executor.get();
}

backend_allocator *vk_backend::get_allocator(device_id dev) const {
  assert(dev.get_backend() == this->get_unique_backend_id());
  return static_cast<vk_hardware_context *>(
             get_hardware_manager()->get_device(dev.get_id()))
      ->get_allocator();
}

std::string vk_backend::get_name() const { return "Vulkan"; }

std::unique_ptr<backend_executor>
vk_backend::create_inorder_executor(device_id dev, int) {
  std::unique_ptr<inorder_queue> q =
      std::make_unique<vk_queue>(&_hw_manager, dev.get_id());

  return std::make_unique<inorder_executor>(std::move(q));
}

std::shared_ptr<dag_node_event>
vk_backend::create_event_from_native_handle(const void *native_handle,
                                            device_id dev) {
  if (!native_handle || dev.get_backend() != backend_id::vk) {
    return nullptr;
  }

  if (dev.get_id() < 0 ||
      static_cast<std::size_t>(dev.get_id()) >= _hw_manager.get_num_devices()) {
    return nullptr;
  }

  const auto *handle =
      static_cast<const vk_event_native_handle *>(native_handle);
  if (handle->semaphore == VK_NULL_HANDLE) {
    return nullptr;
  }

  auto *hw_ctx =
      static_cast<vk_hardware_context *>(_hw_manager.get_device(dev.get_id()));
  const vk::Device device = *hw_ctx->get_device();
  // The semaphore belongs to another device
  if (handle->device != VK_NULL_HANDLE &&
      handle->device != static_cast<VkDevice>(device)) {
    return nullptr;
  }

  vk_event_native_handle imported = *handle;
  imported.device = static_cast<VkDevice>(device);
  return std::make_shared<vk_imported_node_event>(device, imported);
}
} // namespace rt
} // namespace hipsycl
