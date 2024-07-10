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
#ifndef HIPSYCL_HIP_INSTRUMENTATION_HPP
#define HIPSYCL_HIP_INSTRUMENTATION_HPP

#include "hip_event.hpp"
#include "../generic/host_timestamped_event.hpp"
#include "../generic/timestamp_delta_instrumentation.hpp"
#include "../instrumentation.hpp"
#include "hipSYCL/runtime/event.hpp"
#include <chrono>

namespace hipsycl {
namespace rt {

class hip_event_time_delta {
public:
  profiler_clock::duration operator()(const dag_node_event& t0,
                                      const dag_node_event& t1) const;
};

using hip_submission_timestamp = simple_submission_timestamp;

using hip_execution_start_timestamp =
    timestamp_delta_instrumentation<instrumentations::execution_start_timestamp,
                                    hip_event_time_delta>;

using hip_execution_finish_timestamp =
    timestamp_delta_instrumentation<instrumentations::execution_finish_timestamp,
                                    hip_event_time_delta>;

}
}

#endif
