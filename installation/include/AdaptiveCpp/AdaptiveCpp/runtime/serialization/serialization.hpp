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
#ifndef HIPSYCL_DUMP_RUNTIME_HPP
#define HIPSYCL_DUMP_RUNTIME_HPP

#define HIPSYCL_DUMP_INDENTATION "   "
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/util.hpp"

#include <ostream>
#include <sstream>
#include <map>

namespace hipsycl::rt {
std::ostream &operator<<(std::ostream &out, const hardware_platform value);
std::ostream &operator<<(std::ostream &out, const api_platform value);
std::ostream &operator<<(std::ostream &out, const backend_id value);
std::ostream &operator<<(std::ostream &out, device_id dev);

template <int Dim>
std::ostream &operator<<(std::ostream &out, const static_array<Dim> &v) {
  out << "{";
  for (int i = 0; i < Dim; ++i) {
    out << v[i];
    if (i != Dim - 1)
      out << ", ";
  }
  out << "}";
  return out;
}

template <typename T> std::string dump(T *val) {
  std::stringstream sstr;
  val->dump(sstr);
  return sstr.str();
}

} // namespace hipsycl::rt

#endif
