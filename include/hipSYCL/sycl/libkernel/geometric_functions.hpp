/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 hipSYCL contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_GEOMETRIC_FUNCTIONS_HPP
#define HIPSYCL_GEOMETRIC_FUNCTIONS_HPP

#include <type_traits>

#include "common_functions.hpp"

namespace hipsycl {
namespace sycl {

template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline vec<DT, 3> cross(vec<DT, 3> a, vec<DT, 3> b) {
  return {a.y() * b.z() - a.z() * b.y(),
          a.z() * b.x() - a.x() * b.z(),
          a.x() * b.y() - a.y() * b.x()};
}
template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline vec<DT, 4> cross(vec<DT, 4> a, vec<DT, 4> b) {
  auto c =
      cross(vec<DT, 3>{a.x(), a.y(), a.z()}, vec<DT, 3>{b.x(), b.y(), b.z()});
  return {c.x(), c.y(), c.z(), DT{0}};
}

template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline DT dot(DT a, DT b) {
  return a * b;
}
template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline DT dot(vec<DT, 2> a, vec<DT, 2> b) {
  return a.x() * b.x() + a.y() * b.y();
}
template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline DT dot(vec<DT, 3> a, vec<DT, 3> b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z();
}
template<typename DT, HIPSYCL_ENABLE_IF_FLOATING_POINT(DT)>
HIPSYCL_KERNEL_TARGET inline DT dot(vec<DT, 4> a, vec<DT, 4> b) {
  return a.x() * b.x() + a.y() * b.y() + a.z() * b.z() + a.w() * b.w();
}

template<typename T, std::enable_if_t<traits::is_geo_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto length(T v) {
  return sqrt(dot(v, v));
}

template<typename T, std::enable_if_t<traits::is_geo_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto distance(T p0, T p1) {
  return length(p0 - p1);
}

template<typename T, std::enable_if_t<traits::is_geo_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto normalize(T v) {
  return v / length(v);
}

// Fast* functions simply map to standard implementations for now

template<typename T, std::enable_if_t<traits::is_gengeofloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto fast_distance(T p0, T p1) {
  return distance(p0, p1);
}

template<typename T, std::enable_if_t<traits::is_gengeofloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto fast_length(T v) {
  return length(v);
}

template<typename T, std::enable_if_t<traits::is_gengeofloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET inline auto fast_normalize(T v) {
  return normalize(v);
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_GEOMETRIC_FUNCTIONS_HPP
