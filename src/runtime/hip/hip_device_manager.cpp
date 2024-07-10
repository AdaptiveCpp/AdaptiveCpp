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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"

namespace hipsycl {
namespace rt {

hip_device_manager::hip_device_manager() {
  auto err = hipGetDevice(&_device);

  if (err != hipSuccess){
    register_error(
        __acpp_here(),
        error_info{
            "hip_device_manager: Could not obtain currently active HIP device",
            error_code{"HIP", err}});
  }
}

void hip_device_manager::activate_device(int device_id)
{
  if (_device != device_id) {

    HIPSYCL_DEBUG_INFO << "hip_device_manager: Switchting to device "
                       << device_id << std::endl;

    auto err = hipSetDevice(device_id);

    if (err != hipSuccess){
    register_error(
        __acpp_here(),
        error_info{
            "hip_device_manager: Could not set active HIP device",
            error_code{"HIP", err}});
    } else {
      _device = device_id;
    }
  }
}

int hip_device_manager::get_active_device() const
{
  return _device;
}

}
}
