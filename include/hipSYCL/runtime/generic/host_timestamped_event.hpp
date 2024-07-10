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
#ifndef HIPSYCL_HOST_TIMESTAMPED_EVENT_HPP
#define HIPSYCL_HOST_TIMESTAMPED_EVENT_HPP

#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/instrumentation.hpp"
#include "hipSYCL/runtime/event.hpp"

namespace hipsycl {
namespace rt {

class host_timestamped_event {
public:
  host_timestamped_event() = default;

  host_timestamped_event(inorder_queue* q)
  : host_timestamped_event{q->insert_event()} {}

  host_timestamped_event(std::shared_ptr<dag_node_event> evt) 
  : _evt{evt} {
    _evt->wait();
    _time = profiler_clock::now();
  }

  std::shared_ptr<dag_node_event> get_event() const {
    return _evt;
  }

  profiler_clock::time_point get_timestamp() const {
    return _time;
  }
private:
  std::shared_ptr<dag_node_event> _evt;
  profiler_clock::time_point _time;
};


}
}

#endif
