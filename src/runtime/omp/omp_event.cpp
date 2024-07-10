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
#include "hipSYCL/runtime/omp/omp_event.hpp"


namespace hipsycl {
namespace rt {

omp_node_event::omp_node_event()
: _signal_channel{std::make_shared<signal_channel>()}
{}

omp_node_event::~omp_node_event()
{}

bool omp_node_event::is_complete() const {
  return _signal_channel->has_signalled();
}

void omp_node_event::wait() {
  _signal_channel->wait();
}

std::shared_ptr<signal_channel>
omp_node_event::get_signal_channel() const {
  return _signal_channel;
}

std::shared_ptr<signal_channel> omp_node_event::request_backend_event() {
  return get_signal_channel();
}

}
}
