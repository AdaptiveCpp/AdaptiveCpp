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


#include <numeric>
#include <type_traits>

#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(marray_tests, reset_device_fixture)

using marray_test_types = boost::mpl::list<
  float,
  double,
  short,
  unsigned,
  int,
  long long,
  unsigned long long>;

template <class T> T get_input_value1(int i) { return static_cast<T>(i + 1); }
template <class T> T get_input_value2(int i) { return static_cast<T>(2*i + i%3); }

template<class T, int N>
void test(sycl::queue& q) {

  sycl::marray<T,N> a, b;

  for(int i = 0; i < N; ++i) {
    a[i] = get_input_value1<T>(i);
    b[i] = get_input_value2<T>(i);
  }

  constexpr std::size_t num_tests = 4;
  sycl::marray<T, N>* output = sycl::malloc_device<sycl::marray<T,N>>(num_tests, q);

  q.single_task([=](){
    output[0] = a + b;
    output[1] = a - b;
    output[2] = a * b;
    output[3] = a / (b + static_cast<T>(1));
  }).wait();

  std::vector<sycl::marray<T,N>> results(num_tests);
  q.copy(output, results.data(), num_tests).wait();
  sycl::free(output, q);

  auto tolerance = boost::test_tools::tolerance(0.0001);
  // divisions may be less precise with OpenCL backend
  // boost.test seems to not work with double tolerance values
  // when the data type of the test is float :(
  auto get_div_tolerance = [&](){
    if constexpr(std::is_same_v<T, float>) {
      return boost::test_tools::tolerance(0.001f);
    } else {
      return tolerance;
    }
  };
    
  
  for(int i = 0; i < N; ++i) {
    BOOST_TEST(results[0][i] == a[i] + b[i], tolerance);
    BOOST_TEST(results[1][i] == a[i] - b[i], tolerance);
    BOOST_TEST(results[2][i] == a[i] * b[i], tolerance);
    
    T expected_div = a[i] / (b[i] + static_cast<T>(1));
    BOOST_TEST(results[3][i] == expected_div, get_div_tolerance());
  }

}

BOOST_AUTO_TEST_CASE_TEMPLATE(marray_ops, T, marray_test_types::type) {
  sycl::queue q;
  
  test<T, 1>(q);
  test<T, 3>(q);
  test<T, 4>(q);
  test<T, 13>(q);
}

template<class T, std::size_t N>
bool verify_marray_content(const sycl::marray<T,N>& x, const std::array<T,N>& ref) {
  for(int i = 0; i < N; ++i)
    if(x[i] != ref[i])
      return false;
  return true;
}

BOOST_AUTO_TEST_CASE(marray_constructor) {
  sycl::marray<int, 1> v1 {3};
  sycl::marray<int, 3> v2 {1};
  sycl::marray<int, 4> v3 {5, v2};
  sycl::marray<int, 4> v4 {v1, v2};
  sycl::marray<int, 8> v5{v2, v1, 0, v2};

  BOOST_TEST(verify_marray_content(v1, {3}));
  BOOST_TEST(verify_marray_content(v2, {1, 1, 1}));
  BOOST_TEST(verify_marray_content(v3, {5, 1, 1, 1}));
  BOOST_TEST(verify_marray_content(v4, {3, 1, 1, 1}));
  BOOST_TEST(verify_marray_content(v5, {1, 1, 1, 3, 0, 1, 1, 1}));
}

#define MARRAY_ALIAS_SAME(type, storage_type, elements)                 \
  (std::is_same_v<sycl::m##type##elements, sycl::marray<storage_type, elements>>)

#define MARRAY_ALIAS_CHECK(type, storage_type)        \
  MARRAY_ALIAS_SAME(type, storage_type, 2) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 3) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 4) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 8) &&         \
  MARRAY_ALIAS_SAME(type, storage_type, 16)

BOOST_AUTO_TEST_CASE(marray_aliases) {
  BOOST_CHECK(MARRAY_ALIAS_CHECK(char, int8_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(uchar, uint8_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(short, int16_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(ushort, uint16_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(int, int32_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(uint, uint32_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(long, int64_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(ulong, uint64_t));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(half, sycl::half));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(float, float));
  BOOST_CHECK(MARRAY_ALIAS_CHECK(double, double));
}

BOOST_AUTO_TEST_CASE(marray_implicit_conversion) {
  constexpr int val = 42;

  const auto f = [val](int x) {
    BOOST_CHECK_EQUAL(x, val);
  };

  sycl::marray<int, 1> arr(val);
  f(arr);
}

#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
// nvc++ seems to have a problem with these tests
BOOST_AUTO_TEST_CASE(marray_constexpr) {
  constexpr sycl::marray arr1{42};
  constexpr sycl::marray<int, 2> arr2{arr1, arr1};
  constexpr sycl::marray<int, 4> arr3{123, arr2, 456};
}
#endif

BOOST_AUTO_TEST_SUITE_END()
