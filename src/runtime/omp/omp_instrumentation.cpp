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

#include "hipSYCL/runtime/omp/omp_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cassert>

namespace hipsycl {
namespace rt {

inline void omp_timestamp_profiler::record(profiler_clock::time_point &event) {
  {
    std::lock_guard lock{_mutex};
    assert(event == profiler_clock::time_point{});
    event = profiler_clock::now();
  }
  _update.notify_all();
}

void omp_timestamp_profiler::record_submit() {
  record(_operation_submitted);
}

void omp_timestamp_profiler::record_start() {
  record(_operation_started);
}

void omp_timestamp_profiler::record_finish() {
  record(_operation_finished);
}

profiler_clock::time_point omp_timestamp_profiler::await_event(event event) const {
  const profiler_clock::time_point *wait_for;
  switch (event) {
    case event::operation_submitted:
      wait_for = &_operation_submitted;
      break;
    case event::operation_started:
      wait_for = &_operation_started;
      break;
    case event::operation_finished:
      wait_for = &_operation_finished;
      break;
  }

  if (event == event::operation_submitted) {
    if (_operation_submitted == profiler_clock::time_point{}) {
      register_error(__hipsycl_here(),
                     error_info{"omp_timestamp_profiler: submission event not recorded"});
    }
    return _operation_submitted;
  }

  std::unique_lock lock{_mutex};
  _update.wait(lock, [&] { return *wait_for != profiler_clock::time_point{}; });
  return *wait_for;
}

std::unique_ptr<omp_timestamp_profiler> omp_timestamp_profiler::make_no_op()
{
  auto p = std::make_unique<omp_timestamp_profiler>();
  p->_operation_finished = p->_operation_started = p->_operation_submitted = profiler_clock::now();
  return p;
}

}
}

