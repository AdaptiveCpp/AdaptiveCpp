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

#ifndef ACPP_RT_PCUDA_RUNTIME_HPP
#define ACPP_RT_PCUDA_RUNTIME_HPP

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_device_topology.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"

#include <list>
#include <mutex>

namespace hipsycl::rt {

class runtime;

}

namespace hipsycl::rt::pcuda {

class pcuda_runtime {
public:
  pcuda_runtime();

  runtime* get_rt() const {
    return _rt.get();
  }

  const device_topology& get_topology() const {
    return _topology;
  }

private:
  runtime_keep_alive_token _rt;
  device_topology _topology;
};

class pcuda_application {
public:
  static pcuda_application& get() {
    static pcuda_application app;
    return app;
  }

  pcuda_runtime& pcuda_rt();
  const pcuda_runtime& pcuda_rt() const;

  // Note: This function assumes that there is no more
  // then a single pcuda_runtime object at any given time!
  thread_local_state& tls_state();

  pcuda_application(const pcuda_application&) = delete;
  pcuda_application& operator=(const pcuda_application&) = delete;
private:
  pcuda_application() {}

  pcuda_runtime _pcuda_rt;
  // use list to make pointers/iterators stable
  mutable std::list<thread_local_state> _tls_states;
  mutable std::mutex _lock;
};

}

#endif
