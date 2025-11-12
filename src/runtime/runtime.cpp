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
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

namespace hipsycl {
namespace rt {

runtime::runtime()
: _dag_manager{this}
{
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt launch initiated ********"
                      << std::endl;

  Tracer_utils::tracer_state.initialize_tracer();
}

runtime::~runtime()
{
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt shutdown ********"
                      << std::endl;

  Tracer_utils::tracer_state.run_finalizers();
}



}
}
