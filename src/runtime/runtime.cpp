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
  // Initialize global singletons needed during shutdown before any static
  // object that owns the runtime completes construction, so reverse-order
  // static destruction destroys them after runtime shutdown.
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
