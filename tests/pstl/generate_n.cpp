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
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_generate_n, enable_unified_shared_memory)

template<class Policy>
void test_generate_n(Policy&& pol, std::size_t problem_size) {
  std::vector<int> data(problem_size);

  auto ret = std::generate_n(pol, data.begin(), data.size(),
                []() { return 42; });
  BOOST_CHECK(ret == data.begin() + problem_size);

  std::vector<int> data_host(problem_size);

  std::generate_n(data_host.begin(), data.size(),
                []() { return 42; });

  BOOST_CHECK(data == data_host);
}

BOOST_AUTO_TEST_CASE(par_unseq_negative) {
  std::vector<int> empty;

  auto ret = std::generate_n(std::execution::par_unseq, empty.begin(), -1,
                             []() { return 42; });
  BOOST_CHECK(ret == empty.begin());
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_generate_n(std::execution::par_unseq, 0);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_generate_n(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_generate_n(std::execution::par_unseq, 1000);
}


BOOST_AUTO_TEST_CASE(par_empty) {
  test_generate_n(std::execution::par, 0);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  test_generate_n(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  test_generate_n(std::execution::par, 1000);
}

BOOST_AUTO_TEST_SUITE_END()
