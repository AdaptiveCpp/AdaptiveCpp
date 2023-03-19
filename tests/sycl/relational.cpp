/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
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

#include <boost/mpl/joint_view.hpp>

#include <cmath>
#include <cfloat>

BOOST_FIXTURE_TEST_SUITE(rel_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using rel_test_genfloats = boost::mpl::list<
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
  struct vector_elem {
    using type = T;
  };
  template<typename DT, int D>
  struct vector_elem<vec<DT, D>> {
    using type = DT;
  };
  template<typename T>
  using vector_elem_t = typename vector_elem<T>::type;

  // utility functions for generic testing

  template<typename DT, int D, std::enable_if_t<D<=4, int> = 0>
  auto get_rel_input(cl::sycl::vec<DT, 16> v) {
    return std::get<D>(std::make_tuple(
      v.s0(),
      vec<DT, 1>(v.s0()),
      vec<DT, 2>(v.s0(), v.s1()),
      vec<DT, 3>(v.s0(), v.s1(), v.s2()),
      vec<DT, 4>(v.s0(), v.s1(), v.s2(), v.s3())));
  }
  template<typename DT, int D, std::enable_if_t<D==8, int> = 0>
  auto get_rel_input(cl::sycl::vec<DT, 16> v) {
    return vec<DT, 8>(v.s0(), v.s1(), v.s2(), v.s3(), v.s4(), v.s5(), v.s6(), v.s7());
  }
  template<typename DT, int D, std::enable_if_t<D==16, int> = 0>
  auto get_rel_input(cl::sycl::vec<DT, 16> v) {
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
}

BOOST_AUTO_TEST_CASE_TEMPLATE(rel_genfloat_unary, T,
                              rel_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  using IntType = typename std::conditional_t<std::is_same_v<DT, float>, int32_t, int64_t>;
  using OutType = typename s::detail::builtin_type_traits<T>::template alternative_data_type<IntType>;

  constexpr int FUN_COUNT = 5;

  // build inputs and allocate outputs

  s::queue queue;
  s::buffer<T> in{{1}};
  s::buffer<OutType> out{{FUN_COUNT}};
  {
    auto inputs  = in.template get_access<s::access::mode::write>();
    auto outputs = out.template get_access<s::access::mode::write>();
    inputs[0] = get_rel_input<DT, D>({NAN, INFINITY, INFINITY - INFINITY, 0.0, 0.0/0.0, 1.0/0.0, sqrt(-1), FLT_MIN, FLT_MIN/2.0, DBL_MIN, DBL_MIN/2.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = OutType{IntType{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto inputs  = in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class rel_unary, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::isfinite(inputs[0]);
      outputs[i++] = s::isinf(inputs[0]);
      outputs[i++] = s::isnan(inputs[0]);
      outputs[i++] = s::isnormal(inputs[0]);
      outputs[i++] = s::signbit(inputs[0]);
    });
  });

  // check results

  {
    auto inputs = in.template get_access<s::access::mode::read>();
    auto outputs = out.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == std::isfinite(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::isinf(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::isnan(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::isnormal(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::signbit(comp(inputs[0], c)));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
