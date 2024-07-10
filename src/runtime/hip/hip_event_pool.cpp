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
#include "hipSYCL/runtime/hip/hip_event_pool.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"

namespace hipsycl {
namespace rt {

hip_event_factory::hip_event_factory(int device_id)
: _device_id{device_id} {}

result hip_event_factory::create(hipEvent_t& out) {
  hip_device_manager::get().activate_device(_device_id);

  hipEvent_t evt;
  auto err = hipEventCreate(&evt);
  if(err != hipSuccess) {
    return make_error(
        __acpp_here(),
        error_info{"hip_event_factory: Couldn't create event", error_code{"HIP", err}});
    
  }
  out = evt;
  return make_success();
}

result hip_event_factory::destroy(hipEvent_t evt) {
  auto err = hipEventDestroy(evt);
  if (err != hipSuccess) {
    return make_error(__acpp_here(),
                   error_info{"hip_event_factory: Couldn't destroy event",
                              error_code{"HIP", err}});
  }
  return make_success();
}

hip_event_pool::hip_event_pool(int device_id)
: event_pool<hip_event_factory>{hip_event_factory{device_id}} {}


}
}