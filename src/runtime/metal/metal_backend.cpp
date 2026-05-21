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
#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/runtime/metal/metal_backend.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::metal_backend();
}

static const char *backend_name = "metal";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}

namespace hipsycl {
namespace rt {


metal_backend::metal_backend()
  : _executor([this](){
    return std::make_unique<multi_queue_executor>(*this, [this](device_id dev) {
      return std::unique_ptr<metal_inorder_queue>(_hw.make_queue(dev.get_id()));
    });
  })
{
}


api_platform metal_backend::get_api_platform() const {
  return api_platform::metal;
}
hardware_platform metal_backend::get_hardware_platform() const {
  return hardware_platform::metal;
}
backend_id metal_backend::get_unique_backend_id() const {
  return backend_id::metal;
}

backend_hardware_manager* metal_backend::get_hardware_manager() const {
  return &_hw;
}
backend_executor* metal_backend::get_executor(device_id dev) const {
  if (dev.get_backend() != backend_id::metal) {
    register_error(
      __acpp_here(),
      error_info{
        "Requested device ID does not belong to the Metal backend.",
        error_type::invalid_parameter_error}
    );
    return nullptr;
  }

  return _executor.get();
}
backend_allocator *metal_backend::get_allocator(device_id dev) const {
  if (dev.get_backend() != backend_id::metal) {
    register_error(
      __acpp_here(),
      error_info{
        "Requested device ID does not belong to the Metal backend.",
        error_type::invalid_parameter_error}
    );
    return nullptr;
  }
  return _hw.get_allocator(dev.get_id());
}

std::string metal_backend::get_name() const {
  return "Metal";
}

std::unique_ptr<backend_executor>
metal_backend::create_inorder_executor(device_id dev, int priority) {
  std::unique_ptr<inorder_queue> q(_hw.make_queue(dev.get_id()));
  return std::make_unique<inorder_executor>(std::move(q));
}

metal_backend::~metal_backend() = default;

} // namespace rt
} // namespace hipsycl