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

#include <algorithm>
#include <execution>
#include <pstl/glue_execution_defs.h>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_equal, enable_unified_shared_memory)

template <class Policy, class Generator, class BinaryPred>
void test_equal(Policy&& pol, std::size_t problem_size, Generator gen, BinaryPred p) {
  // std::equal (2)
  std::vector<int> data(problem_size), data1(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);

  for(int i = 0; i < problem_size; ++i)
    data1[i] = gen(i);

  auto ret = std::equal(pol, data.begin(), data.end(), data1.begin());
  auto ret_host = std::equal(data.begin(), data.end(), data1.begin());

  BOOST_CHECK(ret == ret_host);

  // std::equal (4)
  std::vector<int> data2(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data2[i] = gen(i) * 2;

  ret = std::equal(pol, data.begin(), data.end(), data2.begin(), p);
  ret_host = std::equal(data.begin(), data.end(), data2.begin(), p);

  BOOST_CHECK(ret == ret_host);

  // std::equal (6)
  ret = std::equal(pol, data.begin(), data.end(), data1.begin(), data1.end());
  ret_host = std::equal(data.begin(), data.end(), data1.begin(), data1.end());

  BOOST_CHECK(ret == ret_host);

  for(int i = problem_size;  i < problem_size * 2; ++i)
    data1.push_back(gen(i));

  ret = std::equal(pol, data.begin(), data.end(), data1.begin(), data1.end());
  ret_host = std::equal(data.begin(), data.end(), data1.begin(), data1.end());

  BOOST_CHECK(ret == ret_host);

  // std::equal (8)
  ret = std::equal(pol, data.begin(), data.end(), data2.begin(), data2.end(), p);
  ret_host = std::equal(data.begin(), data.end(), data2.begin(), data2.end(), p);

  for(int i = problem_size; i < problem_size * 2; ++i)
    data2.push_back(gen(i) * 2);

  ret = std::equal(pol, data.begin(), data.end(), data2.begin(), data2.end(), p);
  ret_host = std::equal(data.begin(), data.end(), data2.begin(), data2.end(), p);

  BOOST_CHECK(ret == ret_host);
}


template<class Policy>
void empty_tests(Policy&& pol) {
  test_equal(pol, 0, [](int i){return i;},[](int x, int y){return x == y;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_equal(pol, 1, [](int i){return i;},[](int x, int y){return x == y / 2;});
  test_equal(pol, 1, [](int i){return i;},[](int x, int y){return x < y;});
  test_equal(pol, 1, [](int i){return i;},[](int x, int y){return x <= y;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_equal(pol, 1000, [](int i){return i;},[](int x, int y){return x == y / 2;});
  test_equal(pol, 1000, [](int i){return i;},[](int x, int y){return x < y;});
  test_equal(pol, 1000, [](int i){return i;},[](int x, int y){return x <= y;});
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  empty_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  single_element_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  medium_size_tests(std::execution::par_unseq);
}



BOOST_AUTO_TEST_CASE(par_empty) {
  empty_tests(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  single_element_tests(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  medium_size_tests(std::execution::par);
}


BOOST_AUTO_TEST_SUITE_END()
