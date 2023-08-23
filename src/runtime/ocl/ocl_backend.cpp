/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

ocl_backend::ocl_backend()
: _executor(*this, [this](device_id dev){
  return std::make_unique<ocl_queue>(&(this->_hw_manager), dev.get_id());
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
        __hipsycl_here(),
        error_info{
            "ocl_backend: Passed device id from other backend to OpenCL backend"});
    return nullptr;
  }

  return &_executor;
}

backend_allocator *ocl_backend::get_allocator(device_id dev) const {
  if (dev.get_full_backend_descriptor().sw_platform != api_platform::ocl) {
    register_error(
        __hipsycl_here(),
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
