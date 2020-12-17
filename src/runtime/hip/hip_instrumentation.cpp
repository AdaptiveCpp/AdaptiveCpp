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

#include "hipSYCL/runtime/hip/hip_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hip/hip_event.hpp"

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

// precondition: activate device
hip_unique_event async_record_event(hipStream_t stream) {
  auto evt = make_hip_event();
  if (!evt) return nullptr;

  // see cuda_instrumentation.cpp async_record_event()
  auto err = hipEventRecord(evt.get(), stream);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_timestamp_profiler: hipEventRecord() failed",
                              error_code{"HIP", err}});
    return nullptr;
  }

  return evt;
}

profiler_clock::duration wait_and_measure_elapsed_time(hipEvent_t start, hipEvent_t end) {
  if (!end) {
    register_error(__hipsycl_here(), error_info{"hip_timestamp_profiler: event not recorded"});
  }

  auto err = hipEventSynchronize(end);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_timestamp_profiler: hipEventSynchronize() failed",
                              error_code{"HIP", err}});
  }

  float ms = 0.0f;
  err = hipEventElapsedTime(&ms, start, end);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_timestamp_profiler: hipEventElapsedTime() failed",
                              error_code{"HIP", err}});
  }

  return std::chrono::round<profiler_clock::duration>(
      std::chrono::duration<float, std::milli>{ms});
}

}

hip_timestamp_profiler::baseline hip_timestamp_profiler::baseline::record(hipStream_t stream) {
  baseline b;
  b._device_event = async_record_event(stream);

  auto err = hipEventSynchronize(b._device_event.get());
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_queue: hipEventSynchronize() failed",
                              error_code{"HIP", err}});
  }

  b._host_time = profiler_clock::now();
  return b;
}

hip_timestamp_profiler::hip_timestamp_profiler(const baseline *b): _baseline(b) {}

void hip_timestamp_profiler::record_before_operation(hipStream_t stream) {
  assert(_operation_submitted == profiler_clock::time_point{});
  assert(_operation_started == nullptr);
  _operation_submitted = profiler_clock::now();
  _operation_started = async_record_event(stream);
}

void hip_timestamp_profiler::record_after_operation(hipStream_t stream) {
  assert(_operation_finished == nullptr);
  _operation_finished = async_record_event(stream);
}

profiler_clock::time_point hip_timestamp_profiler::await_event(event event) const {
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

  // HIP reports elapsed time as a single-precision float with around 1 µs resolution. Since this
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

