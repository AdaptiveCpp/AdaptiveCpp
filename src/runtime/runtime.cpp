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
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/allocation_tracker.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

runtime::runtime()
: _dag_manager{this}
{
  // Ensure global singletons used during runtime and queue teardown are
  // initialized before any static SYCL object holding a runtime_keep_alive_token
  // completes construction, so they are destroyed after runtime shutdown.
  (void)application::get_settings();
  (void)application::errors();
  (void)application::event_handler_layer();
  allocation_tracker::init();

  HIPSYCL_DEBUG_INFO << "runtime: ******* rt launch initiated ********"
                      << std::endl;
}

runtime::~runtime()
{
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt shutdown ********"
                      << std::endl;
}



}
}
