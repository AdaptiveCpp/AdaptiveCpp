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
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_count_if, enable_unified_shared_memory)

template<class Policy, class UnaryPredicate>
void test_count_if(Policy&& pol, std::size_t problem_size, UnaryPredicate p) {
  std::vector<int> data(problem_size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<int>(i);


  auto reference_result = std::count_if(data.begin(), data.end(), p);
  auto res = std::count_if(pol, data.begin(), data.end(), p);
  
  BOOST_CHECK(res == reference_result);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_count_if(std::execution::par_unseq, 0, [](auto x) { return x < 15; });
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_count_if(std::execution::par_unseq, 1, [](auto x) { return x < 15; });
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_count_if(std::execution::par_unseq, 1000, [](auto x) { return x < 15; });
}

// BOOST_AUTO_TEST_CASE(par_empty) {
//   test_count_if(std::execution::par, 0, 5);
// }

// BOOST_AUTO_TEST_CASE(par_single_element) {
//   test_count_if(std::execution::par, 1, 5);
// }

// BOOST_AUTO_TEST_CASE(par_medium_size) {
//   test_count_if(std::execution::par, 1000, 5);
// }

BOOST_AUTO_TEST_SUITE_END()
