/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/device_selector.hpp"
#include "hipSYCL/sycl/platform.hpp"

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl {
namespace sycl {


device::device(const device_selector &deviceSelector) {
  this->_device_id = deviceSelector.select_device()._device_id;
}

platform device::get_platform() const  {
  // We only have one platform
  return platform{};
}

vector_class<device> device::get_devices(info::device_type deviceType) {

  vector_class<device> result;

  rt::application::get_runtime().backends().for_each_backend(
      [&](rt::backend *b) {
        rt::backend_descriptor bd = b->get_backend_descriptor();
        std::size_t num_devices = b->get_hardware_manager()->get_num_devices();

        for (std::size_t dev = 0; dev < num_devices; ++dev) {
          rt::device_id d_id{bd, static_cast<int>(dev)};

          device d;
          d._device_id = d_id;
          
          if (deviceType == info::device_type::all ||
              (deviceType == info::device_type::accelerator && d.is_accelerator()) ||
              (deviceType == info::device_type::cpu && d.is_cpu()) ||
              (deviceType == info::device_type::host && d.is_cpu()) ||
              (deviceType == info::device_type::gpu && d.is_gpu())) {
            
            result.push_back(d);
          }
        }
  });
  
  return result;
}

int device::get_num_devices()
{
  return get_devices(info::device_type::all).size();
}


}
}
