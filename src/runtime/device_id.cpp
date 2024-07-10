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
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/backend.hpp"

namespace hipsycl {
namespace rt {


device_id::device_id(backend_descriptor b, int id)
: _backend{b}, _device_id{id}
{}

bool device_id::is_host() const
{
  return _backend.hw_platform == hardware_platform::cpu;
}

backend_id device_id::get_backend() const {
  return _backend.id;
}

int device_id::get_id() const {
  return _device_id;
}

backend_descriptor device_id::get_full_backend_descriptor() const {
  return _backend;
}

}
}
