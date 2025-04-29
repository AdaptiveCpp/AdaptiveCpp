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
#ifndef HIPSYCL_DUMP_INTERFACE_HPP
#define HIPSYCL_DUMP_INTERFACE_HPP

#include "hipSYCL/sycl.hpp"
#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/item.hpp"
#include "hipSYCL/sycl/range.hpp"

#include <ostream>
#include <map>

// The << operator is implemented for the sycl interface classes
// and enums. For runtime classes and structs a dump member function
// is implemented
namespace hipsycl::sycl {

template <int dimensions>
std::ostream &operator<<(std::ostream &out, const id<dimensions> id)
{
  out << "(";
  for (int i = 0; i < dimensions - 1; i++) {
    out << id[i] << ',';
  }
  out << id[dimensions - 1] << ")";
  return out;
}

template <int dimensions>
std::ostream &operator<<(std::ostream &out, const range<dimensions> range)
{
  out << "(";
  for (int i = 0; i < dimensions - 1; i++) {
    out << range[i] << ',';
  }
  out << range[dimensions - 1] << ")";
  return out;
}

} // namespace hipsycl::sycl
#endif
