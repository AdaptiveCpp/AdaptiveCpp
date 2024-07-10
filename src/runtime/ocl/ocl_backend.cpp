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

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/ocl/ocl_backend.hpp"
#include "hipSYCL/runtime/ocl/ocl_hardware_manager.hpp"
#include "hipSYCL/runtime/ocl/ocl_queue.hpp"
#include <memory>

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::ocl_backend();
}

static const char *backend_name = "ocl";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}

namespace hipsycl {
namespace rt {

namespace {

std::unique_ptr<multi_queue_executor>
create_multi_queue_executor(ocl_backend *b, ocl_hardware_manager* mgr) {
  return std::make_unique<multi_queue_executor>(*b, [b, mgr](device_id dev) {
    return std::make_unique<ocl_queue>(mgr, static_cast<std::size_t>(dev.get_id()));
  });
}

}

ocl_backend::ocl_backend()
: _executor([this](){
  return create_multi_queue_executor(this, &_hw_manager);
}) {}

ocl_backend::~ocl_backend(){}

api_platform ocl_backend::get_api_platform() const {
  return api_platform::ocl;
}

hardware_platform ocl_backend::get_hardware_platform() const {
  return hardware_platform::ocl;
}

backend_id ocl_backend::get_unique_backend_id() const {
  return backend_id::ocl;
}

backend_hardware_manager* ocl_backend::get_hardware_manager() const {
  return &_hw_manager;
}

backend_executor *ocl_backend::get_executor(device_id dev) const {
  if (dev.get_full_backend_descriptor().sw_platform != api_platform::ocl) {
    register_error(
        __acpp_here(),
        error_info{
            "ocl_backend: Passed device id from other backend to OpenCL backend"});
    return nullptr;
  }

  return _executor.get();
}

backend_allocator *ocl_backend::get_allocator(device_id dev) const {
  if (dev.get_full_backend_descriptor().sw_platform != api_platform::ocl) {
    register_error(
        __acpp_here(),
        error_info{
            "ocl_backend: Passed device id from other backend to OpenCL backend"});
    return nullptr;
  }

  ocl_hardware_context *dev_ctx =
      static_cast<ocl_hardware_context *>(_hw_manager.get_device(dev.get_id()));
  return dev_ctx->get_allocator();
}

std::string ocl_backend::get_name() const {
  return "OpenCL";
}

std::unique_ptr<backend_executor>
ocl_backend::create_inorder_executor(device_id dev, int priority) {
  std::unique_ptr<inorder_queue> q =
      std::make_unique<ocl_queue>(&_hw_manager, dev.get_id());

  return std::make_unique<inorder_executor>(std::move(q));
}

}
}
