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
#include "hipSYCL/runtime/cuda/cuda_event_pool.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <cuda_runtime_api.h>

namespace hipsycl {
namespace rt {

cuda_event_factory::cuda_event_factory(int device_id)
: _device_id{device_id} {}

result cuda_event_factory::create(cudaEvent_t& out) {
  cuda_device_manager::get().activate_device(_device_id);

  cudaEvent_t evt;
  auto err = cudaEventCreate(&evt);
  if(err != cudaSuccess) {
    return make_error(
        __acpp_here(),
        error_info{"cuda_event_factory: Couldn't create event", error_code{"CUDA", err}});
    
  }
  out = evt;
  
  return make_success();
}

result cuda_event_factory::destroy(cudaEvent_t evt) {
  auto err = cudaEventDestroy(evt);
  if (err != cudaSuccess) {
    return make_error(__acpp_here(),
                   error_info{"cuda_event_factory: Couldn't destroy event",
                              error_code{"CUDA", err}});
  }
  return make_success();
}

cuda_event_pool::cuda_event_pool(int device_id)
: event_pool<cuda_event_factory>{cuda_event_factory{device_id}} {}

}
}