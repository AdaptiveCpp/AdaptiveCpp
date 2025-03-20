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


#ifndef ACPP_PCUDA_VEC_HPP
#define ACPP_PCUDA_VEC_HPP

template<class T, int N>
constexpr inline int __pcuda_vec_alignment() {
  constexpr int desired = sizeof(T)*N;
  if constexpr(N == 3)
    return sizeof(T);
  else if constexpr (desired == 1 || desired == 2 || desired == 4 ||
                     desired == 8)
    return desired;
  else if constexpr(desired >= 16)
    return 16;
  return sizeof(T);
}

#define __PCUDA_DECLARE_VECTOR_T1(T, name)                                     \
  struct name {                                                                \
    T x;                                                                       \
  };                                                                           \
  inline name make_##name(T x) { return name{x}; }

#define __PCUDA_DECLARE_VECTOR_T2(T, name)                                     \
  struct alignas(__pcuda_vec_alignment<T, 2>()) name {                         \
    T x;                                                                       \
    T y;                                                                       \
    name() = default;                                                          \
    name(T a, T b) : x{a}, y{b} {}                                             \
  };                                                                           \
  inline name make_##name(T x, T y) { return name{x, y}; }
#define __PCUDA_DECLARE_VECTOR_T3(T, name)                                     \
  struct alignas(__pcuda_vec_alignment<T, 3>()) name {                         \
    T x;                                                                       \
    T y;                                                                       \
    T z;                                                                       \
    name() = default;                                                          \
    name(T a, T b, T c) : x{a}, y{b}, z{c} {}                                  \
  };                                                                           \
  inline name make_##name(T x, T y, T z) { return name{x, y, z}; }
#define __PCUDA_DECLARE_VECTOR_T4(T, name)                                     \
  struct alignas(__pcuda_vec_alignment<T, 4>()) name {                         \
    T x;                                                                       \
    T y;                                                                       \
    T z;                                                                       \
    T w;                                                                       \
    name() = default;                                                          \
    name(T a, T b, T c, T d) : x{a}, y{b}, z{c}, w{d} {}                       \
  };                                                                           \
  inline name make_##name(T x, T y, T z, T w) { return name{x, y, z, w}; }

#define __PCUDA_DECLARE_VECTOR_TYPES(basetype, prefix)                         \
  __PCUDA_DECLARE_VECTOR_T1(basetype, prefix##1)                               \
  __PCUDA_DECLARE_VECTOR_T2(basetype, prefix##2)                               \
  __PCUDA_DECLARE_VECTOR_T3(basetype, prefix##3)                               \
  __PCUDA_DECLARE_VECTOR_T4(basetype, prefix##4)

__PCUDA_DECLARE_VECTOR_TYPES(char, char)
__PCUDA_DECLARE_VECTOR_TYPES(unsigned char, uchar)
__PCUDA_DECLARE_VECTOR_TYPES(short, short)
__PCUDA_DECLARE_VECTOR_TYPES(unsigned short, ushort)
__PCUDA_DECLARE_VECTOR_TYPES(int, int)
__PCUDA_DECLARE_VECTOR_TYPES(unsigned int, uint)
__PCUDA_DECLARE_VECTOR_TYPES(long, long)
__PCUDA_DECLARE_VECTOR_TYPES(unsigned long, ulong)
__PCUDA_DECLARE_VECTOR_TYPES(long long, longlong)
__PCUDA_DECLARE_VECTOR_TYPES(unsigned long long, ulonglong)
__PCUDA_DECLARE_VECTOR_TYPES(float, float)
__PCUDA_DECLARE_VECTOR_TYPES(double, double)

#endif
