//===- hipsycl/compiler/cbs/MathUtils.hpp - Math helpers --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Includes / Namespace, formatting
//
//===----------------------------------------------------------------------===//


#ifndef RV_UTILS_MATHUTILS_H_
#define RV_UTILS_MATHUTILS_H_

#include <utility>

#ifdef _MSC_VER
#define COMPILER_VSTUDIO
#else
#include <strings.h>
#endif

#ifdef COMPILER_VSTUDIO
#include <intrin.h>

#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanForward)
#endif

template <typename N> static N gcd(N a, N b) {
  if (a == 0)
    return b;
  else if (b == 0)
    return a;

  if (a > b)
    std::swap(a, b);

  while (a) {
    N q = b % a;
    b = a;
    a = q;
  }

  return b;
}

template <typename N> inline int highest_bit(N n);

template <> inline int highest_bit(unsigned int n) {
#ifdef COMPILER_VSTUDIO
  unsigned long mask = n;
  unsigned long index = 0;
  unsigned char isNonZero = _BitScanReverse(&index, mask);
  if (!isNonZero)
    return -1;
  else
    return sizeof(mask) * 8 - index - 1;

#else
  // default to GCC
  if (n == 0)
    return -1;
  return sizeof(n) * 8 - __builtin_clz(n) - 1;
#endif
}

template <typename N> inline int lowest_bit(N n);

template <> inline int lowest_bit(unsigned int n) {
#ifdef COMPILER_VSTUDIO
  unsigned long mask = n;
  unsigned long index = 0;
  unsigned char isNonZero = _BitScanForward(&index, mask);
  if (!isNonZero)
    return -1;
  else
    return index - 1;

#else
  // default to POSIX
  return ffs(n) - 1;
#endif
}

#endif // RV_UTILS_MATHUTILS_H_
