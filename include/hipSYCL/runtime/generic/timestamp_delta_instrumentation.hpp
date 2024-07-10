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
