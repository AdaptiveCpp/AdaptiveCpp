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

#include <memory>
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_device_topology.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"



namespace hipsycl::rt::pcuda {

pcuda_runtime::pcuda_runtime()
: _topology{get_rt()} {}

thread_local_state& pcuda_application::tls_state() {
  thread_local thread_local_state* tls_state_ptr = nullptr;

  if(!tls_state_ptr) {
    std::lock_guard<std::mutex> lock{_lock};
    
    _tls_states.emplace_back(&_pcuda_rt);
    tls_state_ptr = &(_tls_states.back());
  }
  return *tls_state_ptr;
}

pcuda_runtime &pcuda_application::pcuda_rt() { return _pcuda_rt; }
const pcuda_runtime &pcuda_application::pcuda_rt() const { return _pcuda_rt; }

}


