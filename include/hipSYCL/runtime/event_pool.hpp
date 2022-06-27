/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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
