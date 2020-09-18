/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 hipSYCL contributors
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

#ifndef HIPSYCL_COMMON_FUNCTIONS_HPP
#define HIPSYCL_COMMON_FUNCTIONS_HPP

#include <type_traits>

#include "vec.hpp"
#include "math.hpp"

namespace hipsycl {
namespace sycl {

namespace traits {
  template<typename T>
  struct is_genfloat : std::is_floating_point<T> {};
  template<typename E, int N>
  struct is_genfloat<vec<E, N>> {
    static constexpr bool value = std::is_floating_point<E>::value;
  };
  template<typename T>
  constexpr bool is_genfloat_v = is_genfloat<T>::value;

  template<typename T>
  struct is_geo : std::is_floating_point<T> {};
  template<typename E, int N>
  struct is_geo<vec<E, N>> {
    static constexpr bool value = std::is_floating_point<E>::value && N <= 4;
  };
  template<typename T>
  constexpr bool is_geo_v = is_geo<T>::value;

  template<typename T>
  struct is_gengeofloat : std::is_same<T, float> {};
  template<int N>
  struct is_gengeofloat<vec<float, N>> {
    static constexpr bool value = N <= 4;
  };
  template<typename T>
  constexpr bool is_gengeofloat_v = is_gengeofloat<T>::value;
}

template<typename T, std::enable_if_t<traits::is_genfloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto clamp(T x, T minval, T maxval) {
  return fmin(fmax(x, minval), maxval);
}
template<typename T, typename D,
  std::enable_if_t<!std::is_same<D,T>::value && traits::is_genfloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto clamp(T x, D minval, D maxval) {
  return fmin(fmax(x, T{minval}), T{maxval});
}

template<typename T, HIPSYCL_ENABLE_IF_FLOATING_POINT(T)>
HIPSYCL_KERNEL_TARGET
inline T degrees(T radians) {
  return T{180}/T{M_PI} * radians;
}
template<typename T, typename E = typename T::element_type,
  std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T degrees(T radians) {
  return E{180}/E{M_PI} * radians;
}

template<typename T, std::enable_if_t<traits::is_genfloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto max(T x, T y) {
  return fmax(x, y);
}
template<typename T, typename D,
  std::enable_if_t<!std::is_same<D,T>::value && std::is_floating_point<D>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto max(T x, D y) {
  return fmax(x, T{y});
}

template<typename T, std::enable_if_t<traits::is_genfloat_v<T>, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto min(T x, T y) {
  return fmin(x, y);
}
template<typename T, typename D,
  std::enable_if_t<!std::is_same<D,T>::value && std::is_floating_point<D>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline auto min(T x, D y) {
  return fmin(x, T{y});
}

template<typename T, typename D>
HIPSYCL_KERNEL_TARGET
inline auto mix(T x, T y, D a) {
  return x + (y - x) * a;
}

template<typename T, HIPSYCL_ENABLE_IF_FLOATING_POINT(T)>
HIPSYCL_KERNEL_TARGET
inline T radians(T deg) {
  return T{M_PI}/T{180} * deg;
}
template<typename T, typename E = typename T::element_type, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T radians(T deg) {
  return E{M_PI}/E{180} * deg;
}

template<typename T, HIPSYCL_ENABLE_IF_FLOATING_POINT(T)>
HIPSYCL_KERNEL_TARGET
inline T step(T edge, T x) {
  return x < edge ? 0.0 : 1.0;
}
template<typename T, typename E = typename T::element_type, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T step(T edge, T x) {
  return detail::binary_vector_operation(edge, x, [](E _e, E _x) { return step(_e, _x); });
}
template<typename D, typename T, typename E = typename T::element_type,
  std::enable_if_t<std::is_floating_point<D>::value && !std::is_floating_point<T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T step(D edge, T x) {
  detail::transform_vector(x, [edge](E _x) { return step(edge, _x); });
  return x;
}

template<typename T, HIPSYCL_ENABLE_IF_FLOATING_POINT(T)>
HIPSYCL_KERNEL_TARGET
inline T smoothstep(T edge0, T edge1, T x) {
  T t = clamp((x - edge0) / (edge1 - edge0), T{0}, T{1});
  return t * t * (T{3} - T{2} * t);
}
template<typename T, typename E = typename T::element_type>
HIPSYCL_KERNEL_TARGET
inline T smoothstep(T edge0, T edge1, T x) {
  T t = clamp((x - edge0) / (edge1 - edge0), E{0}, E{1});
  return t * t * (E{3} - E{2} * t);
}
template<typename D, typename T, std::enable_if_t<!std::is_same<D,T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T smoothstep(D edge0, D edge1, T x) {
  return smoothstep(T{edge0}, T{edge1}, x);
}

template<typename T, HIPSYCL_ENABLE_IF_FLOATING_POINT(T)>
HIPSYCL_KERNEL_TARGET
inline T sign(T x) {
  if(x > T{0}) return T{1};
  if(x < T{0}) return T{-1};
  if(std::isnan(x)) return 0;
  return x;
}
template<typename T, typename E = typename T::element_type, std::enable_if_t<!std::is_floating_point<T>::value, int> = 0>
HIPSYCL_KERNEL_TARGET
inline T sign(T x) {
  detail::transform_vector(x, [](E e) { return sign(e); });
  return x;
}

}
}

#endif // HIPSYCL_COMMON_FUNCTIONS_HPP
