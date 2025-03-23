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
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_count, enable_unified_shared_memory)

template<class Policy, class T>
void test_count(Policy&& pol, std::size_t problem_size, const T& value) {
  std::vector<int> data(problem_size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = value;


  auto reference_result = std::count(data.begin(), data.end(), value);
  auto res = std::count(pol, data.begin(), data.end(), value);
  
  BOOST_CHECK(res == reference_result);

  BOOST_CHECK(res == problem_size);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_count(std::execution::par_unseq, 0, 15);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_count(std::execution::par_unseq, 1, 15);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_count(std::execution::par_unseq, 1000, 15);
}

BOOST_AUTO_TEST_CASE(par_empty) {
  test_count(std::execution::par, 0, 15);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  test_count(std::execution::par, 1, 15);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  test_count(std::execution::par, 1000, 15);
}

BOOST_AUTO_TEST_SUITE_END()
