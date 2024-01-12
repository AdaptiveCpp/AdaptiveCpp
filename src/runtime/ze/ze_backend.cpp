/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#include <level_zero/ze_api.h>


#include "hipSYCL/runtime/ze/ze_backend.hpp"
#include "hipSYCL/runtime/ze/ze_allocator.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/runtime/device_id.hpp"

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::ze_backend();
}

static const char *backend_name = "ze";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}

namespace hipsycl {
namespace rt {

ze_backend::ze_backend() {
  ze_result_t err = zeInit(0);

  if (err != ZE_RESULT_SUCCESS) {
    print_warning(__hipsycl_here(),
                  error_info{"ze_backend: Call to zeInit() failed",
                            error_code{"ze", static_cast<int>(err)}});
  }

  _hardware_manager = std::make_unique<ze_hardware_manager>();
  for(std::size_t i = 0; i < _hardware_manager->get_num_devices(); ++i) {
    _allocators.push_back(ze_allocator{
        static_cast<ze_hardware_context *>(_hardware_manager->get_device(i)),
        _hardware_manager.get()});
  }

  _executor = std::make_unique<multi_queue_executor>(
      *this, [this](device_id dev) {
        return std::make_unique<ze_queue>(this->_hardware_manager.get(),
                                          dev.get_id());
      });
}

api_platform ze_backend::get_api_platform() const {
  return api_platform::level_zero;
}

hardware_platform ze_backend::get_hardware_platform() const {
  return hardware_platform::level_zero;
}

backend_id ze_backend::get_unique_backend_id() const {
  return backend_id::level_zero;
}
  
backend_hardware_manager* ze_backend::get_hardware_manager() const {
  return _hardware_manager.get();
}

backend_executor* ze_backend::get_executor(device_id dev) const {
  return _executor.get();
}

backend_allocator *ze_backend::get_allocator(device_id dev) const {
  assert(dev.get_id() < _allocators.size());

  return &(_allocators[dev.get_id()]);
}

std::string ze_backend::get_name() const {
  return "Level Zero";
}

std::unique_ptr<backend_executor>
ze_backend::create_inorder_executor(device_id dev, int priority){
  std::unique_ptr<inorder_queue> q =
      std::make_unique<ze_queue>(_hardware_manager.get(), dev.get_id());

  return std::make_unique<inorder_executor>(std::move(q));
}

}
}
