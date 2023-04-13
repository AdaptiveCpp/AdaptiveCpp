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

#include <musa_runtime_api.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/musa/musa_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

musa_device_manager::musa_device_manager() {
  auto err = musaGetDevice(&_device);

  if (err != musaSuccess){
    register_error(
        __hipsycl_here(),
        error_info{
            "musa_device_manager: Could not obtain currently active MUSA device",
            error_code{"MUSA", err}});
  }
}

void musa_device_manager::activate_device(int device_id)
{
  if (_device != device_id) {

    HIPSYCL_DEBUG_INFO << "musa_device_manager: Switchting to device "
                       << device_id << std::endl;

    auto err = musaSetDevice(device_id);

    if (err != musaSuccess){
    register_error(
        __hipsycl_here(),
        error_info{
            "musa_device_manager: Could not set active MUSA device",
            error_code{"MUSA", err}});
    }
    else {
      _device = device_id;
    }
  }
}

int musa_device_manager::get_active_device() const
{
  return _device;
}

}
}
