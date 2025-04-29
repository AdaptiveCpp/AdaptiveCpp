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
#ifndef HIPSYCL_EVENT_POOL_HPP
#define HIPSYCL_EVENT_POOL_HPP

#include <vector>
#include <mutex>
#include "error.hpp"

namespace hipsycl {
namespace rt {

// BackendEventFactory must satisfy the concept:
// - define event_type for native backend type
// - define method to construct event: result create(event_type&)
// - define method to destroy event: result destroy(event_type)
template<class BackendEventFactory>
class event_pool {
public:
  using event_type = typename BackendEventFactory::event_type;

  event_pool(const BackendEventFactory& event_factory)
      : _event_factory{event_factory} {}

  ~event_pool() {
    for(event_type& evt : _available_events) {
      auto err = _event_factory.destroy(evt);
      if(!err.is_success()) {
        register_error(err);
      }
    }
  }

  // Obtain event from pool. Obtained event
  // must be returned to the pool using release_event()
  // when it is no longer needed.
  result obtain_event(event_type& out) {
    std::lock_guard<std::mutex> lock {_mutex};
    if(!_available_events.empty()) {
      out = _available_events.back();
      _available_events.pop_back();
      return make_success();
    }
    return _event_factory.create(out);
  }

  // Return event to pool.
  void release_event(event_type evt) {
    std::lock_guard<std::mutex> lock {_mutex};
    _available_events.push_back(evt);
  }

private:
  BackendEventFactory _event_factory;
  std::vector<event_type> _available_events;
  std::mutex _mutex;
};

}
}

#endif
