/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_SYCL_FUNCTIONAL_HPP
#define HIPSYCL_SYCL_FUNCTIONAL_HPP

#include "backend.hpp"
#include "hipSYCL/sycl/libkernel/builtins.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

template <typename T> struct plus {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T> struct multiplies {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T> struct bit_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template <typename T> struct bit_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template <typename T> struct bit_xor {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <typename T> struct logical_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template <typename T> struct logical_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

namespace details {
template <typename T> struct vector_elem {
  using type = T;
  static constexpr bool int_based =
      std::is_same_v<type, char> || std::is_same_v<type, unsigned char> ||
      std::is_same_v<type, signed char> || std::is_same_v<type, long> ||
      std::is_same_v<type, unsigned long> || std::is_same_v<type, long long> ||
      std::is_same_v<type, unsigned long long>;
  static constexpr bool float_based =
      std::is_same_v<type, float> || std::is_same_v<type, double>;
};
template <typename DT, int D> struct vector_elem<vec<DT, D>> {
  using type = DT;
  static constexpr bool int_based =
      std::is_same_v<type, char> || std::is_same_v<type, unsigned char> ||
      std::is_same_v<type, signed char> || std::is_same_v<type, long> ||
      std::is_same_v<type, unsigned long> || std::is_same_v<type, long long> ||
      std::is_same_v<type, unsigned long long>;
  static constexpr bool float_based =
      std::is_same_v<type, float> || std::is_same_v<type, double>;
};
} // namespace details

template <typename T> struct minimum {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const {

    using base_t = details::vector_elem<T>;
    if constexpr (base_t::float_based) {
      return sycl::fmin(x, y);
    }

    if constexpr (base_t::int_based) {
      return sycl::min(x, y);
    }

    if constexpr (!base_t::float_based && !base_t::int_based) {

      return (x < y) ? x : y;
    }
  }
};

template <typename T> struct maximum {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const {

    using base_t = details::vector_elem<T>;
    if constexpr (base_t::float_based) {
      return sycl::fmax(x, y);
    }

    if constexpr (base_t::int_based) {
      return sycl::max(x, y);
    }
    if constexpr (!base_t::float_based && !base_t::int_based) {
      return (x > y) ? x : y;
    }
  }
};

} // namespace sycl
} // namespace hipsycl

#endif
