/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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
#include "../instrumentation.hpp"

// Note: CUstream_st* == cudaStream_t
struct CUstream_st;

namespace hipsycl {
namespace rt {

class cuda_timestamp_profiler final : public timestamp_profiler
{
public:
  // host and device "timestamps" for converting relative time measurements
  // from cudaEventElapsedTime to absolute profiler_clock times
  class baseline {
  public:
    // precondition: activate device
    static baseline record(CUstream_st *stream);

    CUevent_st *get_device_event() const { return _device_event.get(); }
    profiler_clock::time_point get_host_time() const { return _host_time; }

  private:
    cuda_unique_event _device_event;
    profiler_clock::time_point _host_time;
  };

  // queue_created_device and queue_created_host must represent the same instance in time.
  // queue_create_device must be recorded and synchronized already.
  explicit cuda_timestamp_profiler(const baseline *b);

  // precondition: activate device
  void record_before_operation(CUstream_st *stream);
  void record_after_operation(CUstream_st *stream);

  virtual profiler_clock::time_point await_event(event event) const override;

private:
  const baseline *_baseline;  // shared, owned by the cuda_queue
  profiler_clock::time_point _operation_submitted;
  cuda_unique_event _operation_started;
  cuda_unique_event _operation_finished;
};

}
}

#endif
