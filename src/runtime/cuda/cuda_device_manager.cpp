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
#include <cuda_runtime_api.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

cuda_device_manager::cuda_device_manager() {
  auto err = cudaGetDevice(&_device);

  if (err != cudaSuccess){
    register_error(
        __acpp_here(),
        error_info{
            "cuda_device_manager: Could not obtain currently active CUDA device",
            error_code{"CUDA", err}});
  }
}

void cuda_device_manager::activate_device(int device_id)
{
  if (_device != device_id) {

    HIPSYCL_DEBUG_INFO << "cuda_device_manager: Switchting to device "
                       << device_id << std::endl;

    auto err = cudaSetDevice(device_id);

    if (err != cudaSuccess){
    register_error(
        __acpp_here(),
        error_info{
            "cuda_device_manager: Could not set active CUDA device",
            error_code{"CUDA", err}});
    }
    else {
      _device = device_id;
    }
  }
}

int cuda_device_manager::get_active_device() const
{
  return _device;
}

}
}
