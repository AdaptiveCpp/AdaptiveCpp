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
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/hip/hip_instrumentation.hpp"
#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

profiler_clock::duration
hip_event_time_delta::operator()(const dag_node_event& t0,
                                 const dag_node_event& t1) const {
  assert(t0.is_complete());
  assert(t1.is_complete());

  hipEvent_t t0_evt = cast<const hip_node_event>(&t0)->get_event();
  hipEvent_t t1_evt = cast<const hip_node_event>(&t1)->get_event();
  
  float ms = 0.0f;
  hipError_t err = hipEventElapsedTime(&ms, t0_evt, t1_evt);

  if (err != hipSuccess) {
    register_error(
        __acpp_here(),
        error_info{"hip_event_time_delta: hipEventElapsedTime() failed",
                   error_code{"HIP", err}});
  }

  return std::chrono::round<profiler_clock::duration>(
      std::chrono::duration<float, std::milli>{ms});
}

}
}

