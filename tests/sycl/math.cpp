/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#include "sycl_test_suite.hpp"

#include <bitset>
#include <boost/mpl/joint_view.hpp>

#include <cmath>

BOOST_FIXTURE_TEST_SUITE(math_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using math_test_genfloats = boost::mpl::list<
  float,
  // vec<T,1> is not genfloat according to SYCL 2020. It's unclear
  // if this is an oversight or intentional.
  //cl::sycl::vec<float, 1>
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<float, 8>,
  cl::sycl::vec<float, 16>,
  double,
  //cl::sycl::vec<double, 1>,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>,
  cl::sycl::vec<double, 8>,
  cl::sycl::vec<double, 16>>;



namespace {

  template<typename DT, int N>
  using vec = cl::sycl::vec<DT, N>;
  auto tolerance = boost::test_tools::tolerance(0.0001);

  // utility type traits for generic testing

  template<typename T>
  struct vector_length {
    static constexpr int value = 0;
  };
  template<typename DT, int N>
  struct vector_length<vec<DT, N>> {
    static constexpr int value = N;
  };
  template<typename T>
  constexpr int vector_length_v = vector_length<T>::value;

  template<typename T>
  struct vector_dim {
    static constexpr int value = 0;
  };
  template<typename DT, int N>
  struct vector_dim<vec<DT, N>> {
    static constexpr int value = 1;
  };
  template<typename T>
  constexpr int vector_dim_v = vector_dim<T>::value;

  template<typename T>
  struct vector_elem {
    using type = T;
  };
  template<typename DT, int D>
  struct vector_elem<vec<DT, D>> {
    using type = DT;
  };
  template<typename T>
  using vector_elem_t = typename vector_elem<T>::type;

  template<typename TARGET_DT, typename T>
  struct vector_coerce_elem {
    using type = TARGET_DT;
  };
  template<typename TARGET_DT, typename DT, int D>
  struct vector_coerce_elem<TARGET_DT, vec<DT, D>> {
    using type = vec<TARGET_DT, D>;
  };
  template<typename TARGET_DT, typename T>
  using vector_coerce_elem_t = typename vector_coerce_elem<TARGET_DT, T>::type;

  // utility functions for generic testing

  template<typename DT, int D, std::enable_if_t<D<=4, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return std::get<D>(std::make_tuple(
      v.s0(),
      vec<DT, 1>(v.s0()),
      vec<DT, 2>(v.s0(), v.s1()),
      vec<DT, 3>(v.s0(), v.s1(), v.s2()),
      vec<DT, 4>(v.s0(), v.s1(), v.s2(), v.s3())));
  }
  template<typename DT, int D, std::enable_if_t<D==8, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return vec<DT, 8>(v.s0(), v.s1(), v.s2(), v.s3(), v.s4(), v.s5(), v.s6(), v.s7());
  }
  template<typename DT, int D, std::enable_if_t<D==16, int> = 0>
  auto get_math_input(cl::sycl::vec<DT, 16> v) {
    return v;
  }

  // runtime indexed access to vector elements
  // this could be a single function with constexpr if in C++17
  // could also be done by using knowledge of the internal structure,
  // but I wanted to avoid that.
  template<typename T, std::enable_if_t<vector_length_v<T> == 0, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx == 0);
    return v;
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 1, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    return v.x();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 2, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    return v.y();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 3, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    if(idx==1) return v.y();
    return v.z();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 4, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.x();
    if(idx==1) return v.y();
    if(idx==2) return v.z();
    return v.w();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 8, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.s0();
    if(idx==1) return v.s1();
    if(idx==2) return v.s2();
    if(idx==3) return v.s3();
    if(idx==4) return v.s4();
    if(idx==5) return v.s5();
    if(idx==6) return v.s6();
    if(idx==7) return v.s7();
    return v.s7();
  }
  template<typename T, std::enable_if_t<vector_length_v<T> == 16, int> = 0>
  auto comp(T v, size_t idx) {
    assert(idx < vector_length_v<T>);
    if(idx==0) return v.s0();
    if(idx==1) return v.s1();
    if(idx==2) return v.s2();
    if(idx==3) return v.s3();
    if(idx==4) return v.s4();
    if(idx==5) return v.s5();
    if(idx==6) return v.s6();
    if(idx==7) return v.s7();
    if(idx==8) return v.s8();
    if(idx==9) return v.s9();
    if(idx==10) return v.sA();
    if(idx==11) return v.sB();
    if(idx==12) return v.sC();
    if(idx==13) return v.sD();
    if(idx==14) return v.sE();
    if(idx==15) return v.sF();
    return v.sF();
  }

  // reference functions

  double ref_clamp(double a, double min, double max) { // in C++17 <algorithm>, remove on upgrade
    if(a < min) return min;
    if(a > max) return max;
    return a;
  }

  static constexpr double pi = 3.1415926535897932385;
  double ref_degrees(double v) {
    return 180.0/pi * v;
  }
  double ref_radians(double v) {
    return pi/180.0 * v;
  }

  double ref_mix(double x, double y, double a) {
    return x + (y - x) * a;
  }

  double ref_step(double edge, double x) {
    if(x < edge) return 0.0;
    return 1.0;
  }

  double ref_smoothstep(double edge0, double edge1, double x) {
    BOOST_REQUIRE(edge0 < edge1); // Standard: results are undefined if edge0 >= edge1
    if(x <= edge0) return 0.0;
    if(x >= edge1) return 1.0;
    double t = ref_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double ref_sign(double x) {
    if(x > 0.0) return 1.0;
    if(x < 0.0) return -1.0;
    if(std::isnan(x)) return 0.0;
    return x;
  }

  template<typename DT, int N>
  DT ref_dot(vec<DT, N> a, vec<DT, N> b) {
    DT ret = DT{0};
    for(int c = 0; c < N; ++c) {
      ret += comp(a, c) * comp(b, c);
    }
    return ret;
  }
  double ref_dot(double a, double b) {
    return a * b;
  }

  double ref_length(double v) {
    return std::abs(v);
  }
  template<typename DT>
  DT ref_length(vec<DT, 2> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y());
  }
  template<typename DT>
  DT ref_length(vec<DT, 3> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
  }
  template<typename DT>
  DT ref_length(vec<DT, 4> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z() + v.w()*v.w());
  }

  template<typename T>
  auto ref_distance(T a, T b) {
    return ref_length(a - b);
  }

  template<typename T>
  auto ref_normalize(T v) {
    return v / ref_length(v);
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_clz(T x) noexcept {
    if(x==0){return sizeof(T)*CHAR_BIT;}
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    int idx = 0;
    while(!bset[sizeof(T)*CHAR_BIT - idx -1]){idx++;}
    return idx;
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_binary, T,
                              math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 8;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class math_binary, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::atan2(acc[0], acc[1]);
      acc[i++] = s::copysign(acc[0], acc[1]);
      acc[i++] = s::fmin(acc[0], acc[1]);
      acc[i++] = s::fmax(acc[0], acc[1]);
#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX
      // This triggers ICE in nvc++, no workaround yet.
      acc[i++] = s::fmod(acc[0], acc[1]);
#endif
      acc[i++] = s::fdim(acc[0], acc[1]);
      acc[i++] = s::hypot(acc[0], acc[1]);
      acc[i++] = s::pow(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::atan2(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::copysign(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fmin(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fmax(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX
      BOOST_TEST(comp(acc[i++], c) == std::fmod(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
#endif
      BOOST_TEST(comp(acc[i++], c) == std::fdim(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::hypot(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::pow(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(common_functions, T,
    math_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 23;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  constexpr DT input_scalar = 3.5f;
  constexpr DT mix_input_1 = 0.5f;
  constexpr DT mix_input_2 = 0.8f;
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions
  // some of these are tested multiple times to ensure that all overloads are covered
  // (e.g. combinations of vec and scalar input)

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class common_functions, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fabs(acc[0]);
      acc[i++] = s::clamp(acc[0], acc[1], acc[1] + static_cast<DT>(10));
      acc[i++] = s::clamp(acc[0], input_scalar, static_cast<DT>(input_scalar + 10));
      acc[i++] = s::degrees(acc[0]);
      acc[i++] = s::fma(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mad(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], input_scalar);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::min(acc[0], input_scalar);
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_2});
      acc[i++] = s::mix(acc[0], acc[1], mix_input_1);
      acc[i++] = s::radians(acc[0]);
      acc[i++] = s::step(acc[0], acc[1]);
      acc[i++] = s::step(input_scalar, acc[0]);
      acc[i++] = s::smoothstep(acc[0], acc[0] + static_cast<DT>(10), acc[1]);
      acc[i++] = s::smoothstep(input_scalar, input_scalar + 1, acc[0]);
      acc[i++] = s::sign(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), comp(acc[1], c), comp(acc[1], c) + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), input_scalar, input_scalar + 10), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_degrees(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance); // mad
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), input_scalar), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_2), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_radians(comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(comp(acc[0], c), comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_step(input_scalar, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(comp(acc[0], c), comp(acc[0], c) + 10, comp(acc[1], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(input_scalar, input_scalar + 1, comp(acc[0], c)), tolerance);
      BOOST_TEST(comp(acc[i++], c) == ref_sign(comp(acc[0], c)), tolerance);
    }
  }
}

// some subset of types classified as "geninteger" in SYCL
using math_test_genints = boost::mpl::list<
  int,
  cl::sycl::vec<int, 2>,
  cl::sycl::vec<int, 3>,
  cl::sycl::vec<int, 16>,
  short,
  cl::sycl::vec<short, 4>,
  unsigned char,
  cl::sycl::vec<unsigned char, 3>,
  unsigned long,
  cl::sycl::vec<unsigned long, 8>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(builtin_int_basic, T, math_test_genints::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 4;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7, -8, 9, -1, 17, -4, -2, 3, 7, -8, 9, -1, 17, -4, -2, 3});
    acc[1] = get_math_input<DT, D>({17, -4, -2, 3, 7, -8, 9, -1, 17, -4, -2, 3, 7, -8, 9, -1});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class builtin_int_basic, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::abs(acc[0]);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::clz(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      if constexpr(std::is_signed<DT>::value)
        BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)));
      else
        BOOST_TEST(comp(acc[i++], c) == comp(acc[0], c));
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_clz(comp(acc[0], c)));
    }
  }
}


// types allowed for the "cross" function
using math_test_crossinputs = boost::mpl::list<
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>>;

BOOST_AUTO_TEST_CASE_TEMPLATE(geometric_cross, T, math_test_crossinputs::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 1;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for (int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric_cross, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::cross(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    int i = 2;
    const auto& res = acc[i++], a = acc[0], b = acc[1];
    BOOST_TEST(res.x() == a.y()*b.z() - a.z()*b.y());
    BOOST_TEST(res.y() == a.z()*b.x() - a.x()*b.z());
    BOOST_TEST(res.z() == a.x()*b.y() - a.y()*b.x());
    if(D==4) BOOST_TEST(comp(res,3) == DT{0});
  }
}

// type classes as per SYCL standard

using math_test_gengeofloats = boost::mpl::list<
  float,
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>>;

using math_test_gengeodoubles = boost::mpl::list<
  double,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>>;

using math_test_gengeo = boost::mpl::joint_view<math_test_gengeofloats, math_test_gengeodoubles>;

BOOST_AUTO_TEST_CASE_TEMPLATE(geometric, T, math_test_gengeo::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 4;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::dot(acc[0], acc[1]);
      acc[i++] = s::length(acc[0]);
      acc[i++] = s::distance(acc[0], acc[1]);
      acc[i++] = s::normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto dot_ref_result = ref_dot(acc[0], acc[1]);
    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(dot_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fast_geometric, T, math_test_gengeofloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  constexpr int FUN_COUNT = 3;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<cl::sycl::access::mode::write>();
    acc[0] = get_math_input<DT, D>({7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    acc[1] = get_math_input<DT, D>({17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class fast_geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fast_length(acc[0]);
      acc[i++] = s::fast_distance(acc[0], acc[1]);
      acc[i++] = s::fast_normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result), tolerance);
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)), tolerance);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
