/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
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

#ifndef TESTS_GROUP_FUNCTIONS_HH
#define TESTS_GROUP_FUNCTIONS_HH

#include <cstddef>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <math.h>
#include <type_traits>

#include <sstream>
#include <string>

using namespace cl;

#ifndef __HIPSYCL_ENABLE_SPIRV_TARGET__
#define HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS
#endif

#ifdef TESTS_GROUPFUNCTION_FULL
using test_types = boost::mpl::list<char, int, unsigned int, long long, float, double,
    sycl::vec<int, 1>, sycl::vec<int, 2>, sycl::vec<int, 3>, sycl::vec<int, 4>, sycl::vec<int, 8>,
    sycl::vec<short, 16>, sycl::vec<long, 3>, sycl::vec<unsigned int, 3>>;
#else
using test_types = boost::mpl::list<char, unsigned int, float, double, sycl::vec<int, 2>>;
#endif

namespace detail {

template<int Line, typename T>
class test_kernel;

template<typename T>
using elementType = std::remove_reference_t<decltype(T{}.s0())>;

template<typename T, int N>
std::string type_to_string(sycl::vec<T, N> v) {
  std::stringstream ss{};

  ss << "(";
  if constexpr (1 <= N)
    ss << +v.s0();
  if constexpr (2 <= N)
    ss << ", " << +v.s1();
  if constexpr (3 <= N)
    ss << ", " << +v.s2();
  if constexpr (4 <= N)
    ss << ", " << +v.s3();
  if constexpr (8 <= N) {
    ss << ", " << +v.s4();
    ss << ", " << +v.s5();
    ss << ", " << +v.s6();
    ss << ", " << +v.s7();
  }
  if constexpr (16 <= N) {
    ss << ", " << +v.s8();
    ss << ", " << +v.s9();
    ss << ", " << +v.sA();
    ss << ", " << +v.sB();
    ss << ", " << +v.sC();
    ss << ", " << +v.sD();
    ss << ", " << +v.sE();
    ss << ", " << +v.sF();
  }
  ss << ")";

  return ss.str();
}

template<typename T>
std::string type_to_string(T x) {
  std::stringstream ss{};
  ss << +x;

  return ss.str();
}

template<typename T, int N>
bool compare_type(sycl::vec<T, N> v1, sycl::vec<T, N> v2) {
  bool ret = true;
  if constexpr (1 <= N)
    ret &= v1.s0() == v2.s0();
  if constexpr (2 <= N)
    ret &= v1.s1() == v2.s1();
  if constexpr (3 <= N)
    ret &= v1.s2() == v2.s2();
  if constexpr (4 <= N)
    ret &= v1.s3() == v2.s3();
  if constexpr (8 <= N) {
    ret &= v1.s4() == v2.s4();
    ret &= v1.s5() == v2.s5();
    ret &= v1.s6() == v2.s6();
    ret &= v1.s7() == v2.s7();
  }
  if constexpr (16 <= N) {
    ret &= v1.s8() == v2.s8();
    ret &= v1.s9() == v2.s9();
    ret &= v1.sA() == v2.sA();
    ret &= v1.sB() == v2.sB();
    ret &= v1.sC() == v2.sC();
    ret &= v1.sD() == v2.sD();
    ret &= v1.sE() == v2.sE();
    ret &= v1.sF() == v2.sF();
  }

  return ret;
}

template<typename T>
bool compare_type(T x1, T x2) {
  return x1 == x2;
}

} // namespace detail
#endif // TESTS_GROUP_FUNCTIONS_HH
