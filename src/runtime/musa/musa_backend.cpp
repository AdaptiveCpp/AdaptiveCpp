/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include "hipSYCL/runtime/musa/musa_backend.hpp"
#include "hipSYCL/runtime/musa/musa_event.hpp"
#include "hipSYCL/runtime/musa/musa_queue.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"


HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::musa_backend();
}

static const char *backend_name = "musa";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}

namespace hipsycl {
namespace rt {

musa_backend::musa_backend()
    : _hw_manager{get_hardware_platform()},
      _executor{*this, [this](device_id dev) {
                  return std::make_unique<musa_queue>(this, dev);
                }} {}

api_platform musa_backend::get_api_platform() const {
  return api_platform::musa;
}

hardware_platform musa_backend::get_hardware_platform() const {
  return hardware_platform::musa;
}

backend_id musa_backend::get_unique_backend_id() const {
  return backend_id::musa;
}

backend_hardware_manager *musa_backend::get_hardware_manager() const {
  return &_hw_manager;
}

backend_executor *musa_backend::get_executor(device_id dev) const {
  if (dev.get_full_backend_descriptor().sw_platform != api_platform::musa) {
    register_error(
        __hipsycl_here(),
        error_info{"musa_backend: Passed device id from other backend to MUSA backend"});
    return nullptr;
  }

  return &_executor;
}

backend_allocator* musa_backend::get_allocator(device_id dev) const {
  assert(dev.get_backend() == this->get_unique_backend_id());
  return static_cast<musa_hardware_context *>(
             get_hardware_manager()->get_device(dev.get_id()))
      ->get_allocator();
}

musa_event_pool* musa_backend::get_event_pool(device_id dev) const {
  assert(dev.get_backend() == this->get_unique_backend_id());
  return static_cast<musa_hardware_context *>(
             get_hardware_manager()->get_device(dev.get_id()))
      ->get_event_pool();
}

std::string musa_backend::get_name() const {
  return "MUSA";
}

std::unique_ptr<backend_executor>
musa_backend::create_inorder_executor(device_id dev, int priority) {
  std::unique_ptr<inorder_queue> q =
      std::make_unique<musa_queue>(this, dev, priority);

  return std::make_unique<inorder_executor>(std::move(q));
}

}
}
