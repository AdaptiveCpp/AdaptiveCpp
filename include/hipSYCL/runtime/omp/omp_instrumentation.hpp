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

#include "hipSYCL/runtime/signal_channel.hpp"
#include "omp_event.hpp"
#include "../instrumentation.hpp"

namespace hipsycl {
namespace rt {

using omp_submission_timestamp = simple_submission_timestamp;

class omp_execution_start_timestamp
    : public instrumentations::execution_start_timestamp {
public:
  virtual profiler_clock::time_point get_time_point() const override;
  virtual void wait() const override;
  void record_time();
private:
  profiler_clock::time_point _time;
  mutable signal_channel _signal;
};

class omp_execution_finish_timestamp
    : public instrumentations::execution_finish_timestamp {
public:
  virtual profiler_clock::time_point get_time_point() const override;

  virtual void wait() const override;
  void record_time();
private:
  profiler_clock::time_point _time;
  mutable signal_channel _signal;
};

}
}

#endif
