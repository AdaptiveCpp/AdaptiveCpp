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

#include "hipSYCL/runtime/pcuda/pcuda_event.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/event.hpp"

namespace hipsycl::rt::pcuda {

pcudaError_t event::create(event *&out, pcuda_runtime *rt, unsigned int flags) {
  out = new event {};
  out->_evt = nullptr;
  out->_rt = rt;

  return pcudaSuccess;
}

pcudaError_t event::destroy(event *s) {
  if(!s)
    return pcudaErrorInvalidValue;

  delete s;

  return pcudaSuccess;
}

pcudaError_t event::wait() {
  if(_evt)
    _evt->wait();
  return pcudaSuccess;
}

bool event::is_recorded() const {
  return _evt != nullptr;
}

bool event::is_complete() const {
  if(!_evt) {
    return pcudaErrorNotReady;
  }

  return _evt->is_complete();
}

pcudaError_t event::record(inorder_queue* q) {
  auto evt = q->insert_event();
  _dev = q->get_device();
  _evt = evt;
  return pcudaSuccess;
}

dag_node_event* event::get_event() const {
  return _evt.get();
}

dag_node_event* event::get_event(pcudaEvent_t evt) {
  if(!evt)
    return nullptr;
  return evt->get_event();
}


}


