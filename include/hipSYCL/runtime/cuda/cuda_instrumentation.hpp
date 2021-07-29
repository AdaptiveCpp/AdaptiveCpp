/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2021 Aksel Alpay and contributors
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
