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
#ifndef HIPSYCL_METAL_BACKEND_HPP
#define HIPSYCL_METAL_BACKEND_HPP

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"
#include "hipSYCL/runtime/metal/metal_hardware_manager.hpp"

namespace hipsycl {
namespace rt {

class metal_backend : public backend
{
public:
  metal_backend();


  virtual api_platform get_api_platform() const override;
  virtual hardware_platform get_hardware_platform() const override;
  virtual backend_id get_unique_backend_id() const override;

  virtual backend_hardware_manager* get_hardware_manager() const override;
  virtual backend_executor* get_executor(device_id dev) const override;
  virtual backend_allocator *get_allocator(device_id dev) const override;

  virtual std::string get_name() const override;

  virtual ~metal_backend();

  std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;
private:
  mutable metal_hardware_manager _hw;
  mutable lazily_constructed_executor<multi_queue_executor> _executor;
};

} // namespace rt
} // namespace hipsycl

#endif
