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

#ifndef HIPSYCL_TIMESTAMP_DELTA_INSTRUMENTATION_HPP
#define HIPSYCL_TIMESTAMP_DELTA_INSTRUMENTATION_HPP

#include "../event.hpp"
#include "host_timestamped_event.hpp"


namespace hipsycl {
namespace rt {

/// An instrumentation that calculates event timestamps
/// based on a time delta relative to a reference event.
template<
  class InstrumentationType,
  class EventTimeDeltaCalculator
  >
class timestamp_delta_instrumentation
    : public InstrumentationType {
public:
  timestamp_delta_instrumentation(
    const host_timestamped_event& t0,
    std::shared_ptr<dag_node_event> event)
  : _t0{t0}, _event{event} {}

  /// Instead of the basic constructor this calculates the timestamp as
  /// t0 + (t1- t0) + (t_event - t1).
  /// This can allow for more accurate calculations of kernel durations
  /// for long running applications if backends utilize low precision time
  /// encodings. E.g. cuda represents time as single precision float.
  timestamp_delta_instrumentation(const host_timestamped_event &t0,
                                  std::shared_ptr<dag_node_event> t1,
                                  std::shared_ptr<dag_node_event> event)
      : _t0{t0}, _t1{t1}, _event{event} {}

  virtual profiler_clock::time_point get_time_point() const override{
    EventTimeDeltaCalculator td;

    assert(_t0.get_event()->is_complete());

    if(!_t1) {
      auto delta = td(*_t0.get_event(), *_event);
    
      return _t0.get_timestamp() + delta;
    } else {
      assert(_t1->is_complete());

      auto delta_t1_t0 = td(*_t0.get_event(), *_t1);
      auto delta_t2_t1 = td(*_t1, *_event);

      return _t0.get_timestamp() + delta_t1_t0 + delta_t2_t1;
    }
  }

  virtual void wait() const override {
    _event->wait();
  }

private:
  host_timestamped_event _t0;
  std::shared_ptr<dag_node_event> _t1;
  std::shared_ptr<dag_node_event> _event;
};


}
}

#endif
