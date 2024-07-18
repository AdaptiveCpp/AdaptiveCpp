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
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_transform_reduce, enable_unified_shared_memory)

template<class Policy, class T>
void test_basic_reduction(Policy&& pol, T init, std::size_t size) {
  std::vector<T> data(size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<T>(i);

  T reference_result = std::transform_reduce(
      data.begin(), data.end(), init, std::plus<>{}, [](auto x) { return x; });
  T res = std::transform_reduce(pol,
      data.begin(), data.end(), init, std::plus<>{}, [](auto x) { return x; });
  
  BOOST_CHECK(res == reference_result);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_basic_reduction(std::execution::par_unseq, 10, 0);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_basic_reduction(std::execution::par_unseq, 10, 1);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_single_work_group) {
  test_basic_reduction(std::execution::par_unseq, 10, 127);
}

BOOST_AUTO_TEST_CASE(par_unseq_int_plus) {
  test_basic_reduction(std::execution::par_unseq, 0, 1000);
}

BOOST_AUTO_TEST_CASE(par_unseq_int_plus_large) {
  test_basic_reduction(std::execution::par_unseq, 0ll, 1000*1000);
}



BOOST_AUTO_TEST_CASE(par_empty) {
  test_basic_reduction(std::execution::par, 10, 0);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  test_basic_reduction(std::execution::par, 10, 1);
}

BOOST_AUTO_TEST_CASE(par_incomplete_single_work_group) {
  test_basic_reduction(std::execution::par, 10, 127);
}

BOOST_AUTO_TEST_CASE(par_int_plus) {
  test_basic_reduction(std::execution::par, 0, 1000);
}

BOOST_AUTO_TEST_CASE(par_int_plus_large) {
  test_basic_reduction(std::execution::par, 0ll, 1000*1000);
}

BOOST_AUTO_TEST_CASE(par_unseq_int_unknown_identity) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;

  int reference_result = std::transform_reduce(
      data.begin(), data.end(), 0, [](auto a, auto b) { return a + b; },
      [](auto x) { return x; });
  int res = std::transform_reduce(
      std::execution::par_unseq, data.begin(), data.end(), 0,
      [](auto a, auto b) { return a + b; }, [](auto x) { return x; });
  BOOST_CHECK(res == reference_result);
}

template<class T>
struct aggregate {
  T a, b;

  friend aggregate<T> operator+(const aggregate &lhs, const aggregate &rhs) {
    return aggregate<T>{lhs.a + rhs.a, lhs.b + rhs.b};
  }
};

BOOST_AUTO_TEST_CASE(par_unseq_aggregate) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;

  auto transform = [](int x){
    return aggregate<int>{x*2, x % 10};
  };

  auto reference_result = std::transform_reduce(
      data.begin(), data.end(), aggregate<int>{0, 0}, std::plus<>{},
      transform);

  auto res =
      std::transform_reduce(std::execution::par_unseq, data.begin(), data.end(),
                            aggregate<int>{0, 0}, std::plus<>{}, transform);
  BOOST_CHECK(res.a == reference_result.a);
  BOOST_CHECK(res.b == reference_result.b);
}

template<class T>
struct aggregate_4 {
  T a, b, c, d;
  [[nodiscard]] constexpr aggregate_4 operator+(const aggregate_4 &that) const { //
    return {a + that.a, b + that.b, c + that.c, d + that.d};
  }
};

BOOST_AUTO_TEST_CASE(par_unseq_aggregate_4_double) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i + 1;

  auto transform = [](int x){
    return aggregate_4<double>{ 1.0*x, 2.0*x, 3.0*x, 4.0*x };
  };

  auto reference_result = std::transform_reduce(
      data.begin(), data.end(), aggregate_4<double>{}, std::plus<>{},
      transform);

  auto res =
      std::transform_reduce(std::execution::par_unseq, data.begin(), data.end(),
                            aggregate_4<double>{ }, std::plus<>{}, transform);

  BOOST_CHECK_EQUAL(res.a,  reference_result.a);
  BOOST_CHECK_EQUAL(res.b,  reference_result.b);
  BOOST_CHECK_EQUAL(res.c,  reference_result.c);
  BOOST_CHECK_EQUAL(res.d,  reference_result.d);
}


BOOST_AUTO_TEST_SUITE_END()
