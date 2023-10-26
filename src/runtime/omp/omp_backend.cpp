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
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/omp/omp_backend.hpp"
#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include <memory>


HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create() {
  return new hipsycl::rt::omp_backend();
}

static const char *backend_name = "omp";

HIPSYCL_PLUGIN_API_EXPORT
const char *hipsycl_backend_plugin_get_name() {
  return backend_name;
}



namespace hipsycl {
namespace rt {

namespace {

std::unique_ptr<inorder_queue> make_omp_queue(device_id dev) {
  return std::make_unique<omp_queue>(dev.get_backend());
}

}

omp_backend::omp_backend()
    : _allocator{device_id{
          backend_descriptor{omp_backend::get_hardware_platform(), omp_backend::get_api_platform()}, 0}},
      _hw{},
      _executor(*this, [](device_id dev) -> std::unique_ptr<inorder_queue> {
        return make_omp_queue(dev);
      }) {}

api_platform omp_backend::get_api_platform() const {
  return api_platform::omp;
}

hardware_platform omp_backend::get_hardware_platform() const {
  return hardware_platform::cpu;
}

backend_id omp_backend::get_unique_backend_id() const {
  return backend_id::omp;
}
  
backend_hardware_manager* omp_backend::get_hardware_manager() const {
  return &_hw;
}

backend_executor* omp_backend::get_executor(device_id dev) const {
  if(dev.get_backend() != this->get_unique_backend_id()) {
    register_error(__hipsycl_here(),
                   error_info{"omp_backend: Device id from other backend requested",
                              error_type::invalid_parameter_error});
    return nullptr;
  }

  return &_executor;
}

backend_allocator* omp_backend::get_allocator(device_id dev) const {
  if(dev.get_backend() != this->get_unique_backend_id()) {
    register_error(__hipsycl_here(),
                   error_info{"omp_backend: Device id from other backend requested",
                              error_type::invalid_parameter_error});
    return nullptr;
  }
  return &_allocator;
}

std::string omp_backend::get_name() const {
  return "OpenMP";
}

std::unique_ptr<backend_executor>
omp_backend::create_inorder_executor(device_id dev, int priority){
  return nullptr;
}

}
}
