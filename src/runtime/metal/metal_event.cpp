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

namespace hipsycl {
namespace rt {

metal_node_event::metal_node_event()
  : _signal_channel{std::make_shared<signal_channel>()} {}

metal_node_event::~metal_node_event() {}

bool metal_node_event::is_complete() const {
  return _signal_channel->has_signalled();
}

void metal_node_event::wait() {
  _signal_channel->wait();
}

std::shared_ptr<signal_channel> metal_node_event::get_signal_channel() const {
  return _signal_channel;
}

std::shared_ptr<signal_channel> metal_node_event::request_backend_event() {
  return get_signal_channel();
}

} // namespace rt
} // namespace hipsycl
