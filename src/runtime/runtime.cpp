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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"
#include <mutex>

namespace hipsycl {
namespace rt {

std::atomic<int> runtime::counter = 0;
std::once_flag runtime::init_once;

runtime::runtime() : _dag_manager{this} {
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt launch initiated ********"
                     << std::endl;

  int expected = 0;
  if (counter.compare_exchange_strong(expected, 1))
    std::call_once(init_once,
                   []() { Tracer_utils::initialize_tracers_from_env(); });
  else
    counter.fetch_add(1);
}

runtime::~runtime() {
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt shutdown ********" << std::endl;

  int num = counter.fetch_add(-1) - 1;

  if (num == 0)
    Tracer_utils::finalize_tracing();
}

} // namespace rt
} // namespace hipsycl
