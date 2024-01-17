/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hw_model/hw_model.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

#include <algorithm>

namespace hipsycl {
namespace rt {

backend_manager::backend_manager()
  : _hw_model(std::make_unique<hw_model>(this)),
    _kernel_cache{kernel_cache::get()}
{

  _loader.query_backends();

  for (std::size_t backend_index = 0;
       backend_index < _loader.get_num_backends(); ++backend_index) {

    HIPSYCL_DEBUG_INFO << "Registering backend: '"
                       << _loader.get_backend_name(backend_index) << "'..."
                       << std::endl;
    backend *b = _loader.create(backend_index);
    if (b) {
      _backends.emplace_back(std::unique_ptr<backend>(b));
    } else {
      HIPSYCL_DEBUG_ERROR << "backend_manager: Backend creation failed" << std::endl;
    }
  }
  
  this->for_each_backend([](backend *b) {
    HIPSYCL_DEBUG_INFO << "Discovered devices from backend '" << b->get_name()
                       << "': " << std::endl;
    backend_hardware_manager* hw_manager = b->get_hardware_manager();
    if(hw_manager->get_num_devices() == 0) {
      HIPSYCL_DEBUG_INFO << "  <no devices>" << std::endl;
    } else {
      for(std::size_t i = 0; i < hw_manager->get_num_devices(); ++i){
        hardware_context* hw = hw_manager->get_device(i);

        HIPSYCL_DEBUG_INFO << "  device " << i << ": " << std::endl;
        HIPSYCL_DEBUG_INFO << "    vendor: " << hw->get_vendor_name() << std::endl;
        HIPSYCL_DEBUG_INFO << "    name: " << hw->get_device_name() << std::endl;
      }
    }
  });

  if(std::none_of(_backends.cbegin(), _backends.cend(), 
                  [](const std::unique_ptr<backend>& b){
                    return b->get_hardware_platform() == hardware_platform::cpu;
                    }))
  {
    HIPSYCL_DEBUG_ERROR << "No CPU backend has been loaded. Terminating." << std::endl;
    std::terminate();
  }
}

backend_manager::~backend_manager()
{
  _kernel_cache->unload();
}

backend *backend_manager::get(backend_id id) const {
  auto it = std::find_if(_backends.begin(), _backends.end(),
                         [id](const std::unique_ptr<backend> &b) -> bool {
                           return b->get_backend_descriptor().id == id;
                         });
  
  if(it == _backends.end()){
    register_error(
        __hipsycl_here(),
        error_info{"backend_manager: Requested backend is not available.",
                   error_type::runtime_error});

    return nullptr;
  }
  return it->get();
}

hw_model &backend_manager::hardware_model()
{
  return *_hw_model;
}

const hw_model &backend_manager::hardware_model() const 
{
  return *_hw_model;
}

}
}
