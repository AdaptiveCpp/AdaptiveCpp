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

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"

#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"

namespace hipsycl {
namespace rt {

class vk_backend : public backend {
public:
  vk_backend();
  api_platform get_api_platform() const override;
  hardware_platform get_hardware_platform() const override;
  backend_id get_unique_backend_id() const override;

  backend_hardware_manager *get_hardware_manager() const override;
  backend_executor *get_executor(device_id dev) const override;
  backend_allocator *get_allocator(device_id dev) const override;

  std::string get_name() const override;

  ~vk_backend() {}

  std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;

private:
  mutable vk_hardware_manager _hw_manager;
  mutable lazily_constructed_executor<multi_queue_executor> _executor;
};
} // namespace rt
} // namespace hipsycl
