/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
