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
