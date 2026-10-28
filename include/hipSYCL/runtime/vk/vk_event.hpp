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

#include "../inorder_queue_event.hpp"
#include "hipSYCL/runtime/vk/vk_native_objects.hpp"
#include <vulkan/vulkan.hpp>

namespace hipsycl {
namespace rt {

class vk_queue;

class vk_node_event : public inorder_queue_event<vk::Semaphore>,
                      public vk_native_event_provider {
public:
  using backend_event_type = vk::Semaphore;

  vk_node_event(vk_queue *queue, uint64_t val);

  bool is_complete() const override;
  void wait() override;

  backend_event_type get_event() const;
  device_id get_device() const;

  backend_event_type request_backend_event() override;
  uint64_t get_signal_val() const;

  vk_event_native_handle get_native_event_handle() const override;

private:
  vk_queue *_queue;
  uint64_t _signal_val; // Timeline value of semaphore when submission complete
};

// Event constructed from a timeline semaphore that the runtime does not own,
// used to import synchronization points of an application that submits its own
// work to Vulkan.
class vk_imported_node_event : public inorder_queue_event<vk::Semaphore>,
                               public vk_native_event_provider {
public:
  using backend_event_type = vk::Semaphore;

  vk_imported_node_event(vk::Device device, vk_event_native_handle handle);

  bool is_complete() const override;
  void wait() override;

  backend_event_type request_backend_event() override;

  vk_event_native_handle get_native_event_handle() const override;

private:
  vk::Device _device;
  vk_event_native_handle _handle;
};

} // namespace rt
} // namespace hipsycl
