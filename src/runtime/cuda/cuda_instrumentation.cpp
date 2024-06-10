/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#include "hipSYCL/runtime/cuda/cuda_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"

#include <cuda_runtime_api.h>

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

profiler_clock::duration
cuda_event_time_delta::operator()(const dag_node_event& t0,
                                  const dag_node_event& t1) const {
  assert(t0.is_complete());
  assert(t1.is_complete());

  cudaEvent_t t0_evt = cast<const cuda_node_event>(&t0)->get_event();
  cudaEvent_t t1_evt = cast<const cuda_node_event>(&t1)->get_event();
  
  float ms = 0.0f;
  cudaError_t err = cudaEventElapsedTime(&ms, t0_evt, t1_evt);

  if (err != cudaSuccess) {
    register_error(
        __acpp_here(),
        error_info{"cuda_event_time_delta: cudaEventElapsedTime() failed",
                   error_code{"CUDA", err}});
  }

  return std::chrono::round<profiler_clock::duration>(
      std::chrono::duration<float, std::milli>{ms});
}

}
}

