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


#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(vec_tests, reset_device_fixture)



BOOST_AUTO_TEST_CASE(vec_api) {
  cl::sycl::queue queue;
  cl::sycl::buffer<float, 1> results{72};
  cl::sycl::buffer<float, 1> input{4};

  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = results.get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto inAcc = input.get_access<cl::sycl::access::mode::read_write>(cgh);
    cgh.single_task<class vec_api>([=]() {
      size_t offset = 0;
      const auto store_results = [=, &offset](
        const std::initializer_list<float>& results) {
          for(auto r : results) {
            acc[offset++] = r;
          }
        };

      const cl::sycl::vec<float, 4> v1(1.0f);
      store_results({v1.x(), v1.y(), v1.z(), v1.w()});

      const cl::sycl::vec<float, 8> v2(1.f, 2.f, 4.f, v1, 8.f);
      store_results({v2.s0(), v2.s1(), v2.s2(), v2.s3(),
        v2.s4(), v2.s5(), v2.s6(), v2.s7()});

      // Broadcasting math functions over vector elements
      const auto v3 = cl::sycl::log2(v2);
      store_results({v3.s0(), v3.s1(), v3.s2(), v3.s3(),
        v3.s4(), v3.s5(), v3.s6(), v3.s7()});

      const auto v4 = cl::sycl::fma(v2, v2, v2);
      store_results({v4.s0(), v4.s1(), v4.s2(), v4.s3(),
        v4.s4(), v4.s5(), v4.s6(), v4.s7()});

      const auto v5 = v4 + v4;
      store_results({v5.s0(), v5.s1(), v5.s2(), v5.s3(),
        v5.s4(), v5.s5(), v5.s6(), v5.s7()});

      const auto v6 = cl::sycl::cos(v2) < cl::sycl::sin(v2);
      store_results({(float)v6.s0(), (float)v6.s1(), (float)v6.s2(), (float)v6.s3(),
        (float)v6.s4(), (float)v6.s5(), (float)v6.s6(), (float)v6.s7()});

      // Swizzles!

      // nvc++ currently has a bug that causes it to
      // fail with an LLVM backend error
      // when compiling one of the swizzle tests
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      const cl::sycl::vec<float, 4> v7 = v2.lo();
      store_results({v7.x(), v7.y(), v7.z(), v7.w()});

      const cl::sycl::vec<float, 4> v8 = v7.wzyx();
      store_results({v8.x(), v8.y(), v8.z(), v8.w()});

      cl::sycl::vec<float, 4> v9(1.f);
      v9.xwyz() = v7.xxyy();
      store_results({v9.x(), v9.y(), v9.z(), v9.w()});

      // Nested swizzles
      cl::sycl::vec<float, 4> v10(4.f);
      v10.zxyw().lo() = v7.xxyy().hi();
      store_results({v10.x(), v10.y(), v10.z(), v10.w()});

      // N - n swizzles
      cl::sycl::vec<float, 4> v11(3.f);
      v11.xzw() = v8.xyy();
      store_results({ v11.x(), v11.y(), v11.z(), v11.w() });

      cl::sycl::vec<float, 4> v12(3.f);
      v12.zy() = v8.hi();
      store_results({ v12.x(), v12.y(), v12.z(), v12.w() });
#endif
      cl::sycl::vec<float, 4> v13(12.f);
      v13.store(0, cl::sycl::make_ptr<float, cl::sycl::access::address_space::global_space>(&inAcc[0]));

      cl::sycl::vec<float, 4> v13l;
      v13l.load(0, cl::sycl::make_ptr<const float, cl::sycl::access::address_space::global_space>(&inAcc[0]));
      store_results({ v13l.x(), v13l.y(), v13l.z(), v13l.w() });
    });
  });

  auto acc = results.get_access<cl::sycl::access::mode::read>();
  size_t offset = 0;
  const auto verify_results = [&](const std::initializer_list<float>& expected) {
    for(auto e : expected) {
      BOOST_TEST(acc[offset++] == e);
    }
  };

  verify_results({1.f, 1.f, 1.f, 1.f});                         // v1
  verify_results({1.f, 2.f, 4.f, 1.f, 1.f, 1.f, 1.f, 8.f});     // v2
  verify_results({0.f, 1.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f});     // v3
  verify_results({2.f, 6.f, 20.f, 2.f, 2.f, 2.f, 2.f, 72.f});   // v4
  verify_results({4.f, 12.f, 40.f, 4.f, 4.f, 4.f, 4.f, 144.f}); // v5
  verify_results({1.f, 1.f, 0.f, 1.f, 1.f, 1.f, 1.f, 1.f});     // v6
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  verify_results({1.f, 2.f, 4.f, 1.f});                         // v7
  verify_results({1.f, 4.f, 2.f, 1.f});                         // v8
  verify_results({1.f, 2.f, 2.f, 1.f});                         // v9
  verify_results({2.f, 4.f, 2.f, 4.f});                         // v10
  verify_results({1.f, 3.f, 4.f, 4.f});                         // v11
  verify_results({3.f, 1.f, 2.f, 3.f});                         // v12
#endif
  verify_results({12.f, 12.f, 12.f, 12.f});                     // v13
}


// Regression test: convert<>() would not compile because of illegal private data member access
BOOST_AUTO_TEST_CASE(vec_convert) {
  auto floats_in = cl::sycl::float4{1.f, 2.f, 3.f, 4.f};
  auto ints = floats_in.convert<int>();
  auto floats_out = ints.convert<float>();
  BOOST_TEST(floats_in.x() == floats_out.x());
  BOOST_TEST(floats_in.y() == floats_out.y());
  BOOST_TEST(floats_in.z() == floats_out.z());
  BOOST_TEST(floats_in.w() == floats_out.w());
}


BOOST_AUTO_TEST_SUITE_END()
