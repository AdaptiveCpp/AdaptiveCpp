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


namespace hipsycl {
namespace sycl {


device::device(const device_selector &deviceSelector) {
  this->_device_id = deviceSelector.select_device()._device_id;
}

platform device::get_platform() const  {
  // We only have one platform
  return platform{};
}

vector_class<device> device::get_devices(
    info::device_type deviceType)
{
  if(deviceType == info::device_type::cpu ||
     deviceType == info::device_type::host)
    return vector_class<device>();

  vector_class<device> result;
  int num_devices = get_num_devices();
  for(int i = 0; i < num_devices; ++i)
  {
    device d;
    d._device_id = i;

    result.push_back(d);
  }
  return result;
}

int device::get_num_devices()
{
  int num_devices = 0;
  detail::check_error(hipGetDeviceCount(&num_devices));
  return num_devices;
}

#ifdef HIPSYCL_HIP_INTEROP
int device::get_device_id() const {
  return _device_id;
}
#endif

namespace detail {

void set_device(const device& d)
{
  detail::check_error(hipSetDevice(d._device_id));
}

}

}
}
