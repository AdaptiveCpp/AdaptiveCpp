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
#include "hipSYCL/runtime/metal/metal_event.hpp"

#include <Metal/Metal.hpp>

#include <thread>

namespace hipsycl {
namespace rt {

metal_node_event::metal_node_event(metal_event_handle handle)
  : _handle{handle} {
  _handle.event->retain();
}

metal_node_event::~metal_node_event() {
  _handle.event->release();
}

bool metal_node_event::is_complete() const {
  return _handle.event->signaledValue() >= _handle.value;
}

void metal_node_event::wait() {
  while (!_handle.event->waitUntilSignaledValue(_handle.value, 1000)) {
    std::this_thread::yield();
  }
}

metal_event_handle metal_node_event::request_backend_event() {
  return _handle;
}

} // namespace rt
} // namespace hipsycl
