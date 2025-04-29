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
#ifndef HIPSYCL_INFO_EVENT_HPP
#define HIPSYCL_INFO_EVENT_HPP

#include <cstdint>

#include "info.hpp"
#include "../types.hpp"

namespace hipsycl {
namespace sycl {
namespace info {

enum class event_command_status : int
{
  submitted,
  running,
  complete
};

namespace event
{
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(command_execution_status, event_command_status);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(reference_count, detail::u_int);
};

namespace event_profiling
{
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(command_submit, uint64_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(command_start, uint64_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(command_end, uint64_t);
};

}
}
}

#endif
