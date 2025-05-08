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

namespace hipsycl {
namespace rt {

runtime::runtime() : _dag_manager{this} {
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt launch initiated ********"
                     << std::endl;

  Tracer_utils::initialize_tracers_from_env();

  //  for (int i = 0; i < Tracer_utils::size; i++) {
  //    Tracer_utils::tracer_funcs_array[i](Tracer_utils::PARALLEL_FOR,
  //                                        Tracer_utils::START);
  //  }

  // std::cout << "Hello from hipSYCL!" << std::endl;
}

runtime::~runtime() {
  HIPSYCL_DEBUG_INFO << "runtime: ******* rt shutdown ********" << std::endl;
}

} // namespace rt
} // namespace hipsycl
