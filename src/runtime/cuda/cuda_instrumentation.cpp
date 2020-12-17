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

#include "hipSYCL/runtime/cuda/cuda_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"

#include <cuda_runtime_api.h>

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

// precondition: activate device
cuda_unique_event async_record_event(cudaStream_t stream) {
  auto evt = make_cuda_event();
  if (!evt) return nullptr;

  // cudaEventRecord has different semantics for null and non-null stream parameters. A null-stream
  // will serialize the CUDA context, waiting until all streams have finished their current task
  // before recording the event. A non-null stream will record the event as soon as the previous
  // task in the same stream has retired.
  // There is some discussion online about which type of call should be preferred for timing, e.g.
  // https://stackoverflow.com/a/5846331/1522056 or https://stackoverflow.com/a/49331700/1522056.
  // If a kernel using all GPU resources is running on another stream, passing a null-stream here
  // avoids counting that kernel's remaining runtime towards the measured interval. However, as far
  // as I understand, there is no guarantee for the time between two serializing events: Kernel
  // launches from other threads could race with the kernel launch we intend to measure, and other
  // processes on the machine might also tie up GPU resources from other contexts.
  // Furthermore, serializing the context makes accurately profiling overlapped kernel executions
  // and buffer transfers impossible, which a user might expect to be able to measure.
  // I decided in favor of the non-serializing variant, which has the same semantics that can be
  // expected when measuring CPU execution times on the host with gettimeofday() on a multicore
  // CPU with a preempting process scheduler.
  auto err = cudaEventRecord(evt.get(), stream);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_timestamp_profiler: cudaEventRecord() failed",
                              error_code{"CUDA", err}});
    return nullptr;
  }

  return evt;
}

profiler_clock::duration wait_and_measure_elapsed_time(cudaEvent_t start, cudaEvent_t end) {
  if (!end) {
    register_error(__hipsycl_here(), error_info{"cuda_timestamp_profiler: event not recorded"});
  }

  auto err = cudaEventSynchronize(end);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_timestamp_profiler: cudaEventSynchronize() failed",
                              error_code{"CUDA", err}});
  }

  float ms = 0.0f;
  err = cudaEventElapsedTime(&ms, start, end);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_timestamp_profiler: cudaEventElapsedTime() failed",
                              error_code{"CUDA", err}});
  }

  return std::chrono::round<profiler_clock::duration>(
      std::chrono::duration<float, std::milli>{ms});
}

}

cuda_timestamp_profiler::baseline cuda_timestamp_profiler::baseline::record(cudaStream_t stream) {
  baseline b;
  b._device_event = async_record_event(stream);

  auto err = cudaEventSynchronize(b._device_event.get());
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_queue: cudaEventSynchronize() failed",
                              error_code{"CUDA", err}});
  }

  b._host_time = profiler_clock::now();
  return b;
}

cuda_timestamp_profiler::cuda_timestamp_profiler(const baseline *b): _baseline(b) {}

void cuda_timestamp_profiler::record_before_operation(CUstream_st *stream) {
  assert(_operation_submitted == profiler_clock::time_point{});
  assert(_operation_started == nullptr);
  _operation_submitted = profiler_clock::now();
  _operation_started = async_record_event(stream);
}

void cuda_timestamp_profiler::record_after_operation(CUstream_st *stream) {
  assert(_operation_finished == nullptr);
  _operation_finished = async_record_event(stream);
}

profiler_clock::time_point cuda_timestamp_profiler::await_event(event event) const {
  if (event == event::operation_submitted) {
    // The function creating the profiler must also record the submission. This happens before
    // yielding the profiler to the profiler_promise, so all reads on that variable can happen
    // concurrently without synchronization.
    assert(_operation_submitted != profiler_clock::time_point{});
    return _operation_submitted;
  }

  auto queue_creation_to_start = wait_and_measure_elapsed_time(
      _baseline->get_device_event(), _operation_started.get());
  if (event == event::operation_started) {
    return _baseline->get_host_time() + queue_creation_to_start;
  }

  // CUDA reports elapsed time as a single-precision float with 0.5 µs resolution. Since this
  // type can only represent 7.22 decimal digits, precision will drop once the elapsed time
  // exceeds about 10 seconds (= 1e7 µs). To allow precise measurement of run times (end - start),
  // we measure twice and add the durations in the higher-precision profiler_clock representation.

  assert(event == event::operation_finished);
  auto start_to_end = wait_and_measure_elapsed_time(_operation_started.get(),
                                                    _operation_finished.get());
  return _baseline->get_host_time() + queue_creation_to_start + start_to_end;
}

}
}

