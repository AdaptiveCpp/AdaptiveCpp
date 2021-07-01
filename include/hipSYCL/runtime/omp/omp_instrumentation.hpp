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

#ifndef HIPSYCL_OMP_INSTRUMENTATION_HPP
#define HIPSYCL_OMP_INSTRUMENTATION_HPP

#include <condition_variable>
#include <mutex>

#include "omp_event.hpp"
#include "../instrumentation.hpp"

namespace hipsycl {
namespace rt {

class omp_timestamp_profiler final : public timestamp_profiler
{
public:
  static std::unique_ptr<omp_timestamp_profiler> make_no_op();

  void record_submit();  // not thread-safe
  void record_start();  // thread-safe
  void record_finish();  // thread-safe

  virtual profiler_clock::time_point await_event(event event) const override; // thread-safe

private:
  profiler_clock::time_point _operation_submitted;
  profiler_clock::time_point _operation_started;
  profiler_clock::time_point _operation_finished;
  mutable std::mutex _mutex;
  mutable std::condition_variable _update;

  void record(profiler_clock::time_point &event);
};

}
}

#endif
