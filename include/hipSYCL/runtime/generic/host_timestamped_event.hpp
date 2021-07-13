/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2021 Aksel Alpay and contributors
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
