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
#ifndef HIPSYCL_CUDA_INSTRUMENTATION_HPP
#define HIPSYCL_CUDA_INSTRUMENTATION_HPP

#include "cuda_event.hpp"
#include "../generic/host_timestamped_event.hpp"
#include "../generic/timestamp_delta_instrumentation.hpp"
#include "../instrumentation.hpp"
#include "hipSYCL/runtime/event.hpp"
#include <chrono>

namespace hipsycl {
namespace rt {

class cuda_event_time_delta {
public:
  profiler_clock::duration operator()(const dag_node_event& t0,
                                      const dag_node_event& t1) const;
};

using cuda_submission_timestamp = simple_submission_timestamp;

using cuda_execution_start_timestamp =
    timestamp_delta_instrumentation<instrumentations::execution_start_timestamp,
                                    cuda_event_time_delta>;

using cuda_execution_finish_timestamp =
    timestamp_delta_instrumentation<instrumentations::execution_finish_timestamp,
                                    cuda_event_time_delta>;

}
}

#endif
