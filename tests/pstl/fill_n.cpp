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

BOOST_FIXTURE_TEST_SUITE(pstl_fill_n, enable_unified_shared_memory)


template<class T, class Policy>
void test_fill_n(Policy&& pol, std::size_t problem_size) {
  std::vector<T> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = T{i};
  }

  auto ret =
      std::fill_n(pol, data.begin(), data.size(), T(42));

  BOOST_CHECK(ret == data.begin() + problem_size);

  std::vector<T> host_data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
   host_data[i] = T{i};
  }

  std::fill_n(host_data.begin(), host_data.size(), T(42));

  BOOST_CHECK(host_data == data);
}

BOOST_AUTO_TEST_CASE(par_unseq_negative) {
  std::vector<int> empty;

  auto ret =
      std::fill_n(std::execution::par_unseq, empty.begin(), -1, 42);
  BOOST_CHECK(ret == empty.begin());
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 1000);
}


BOOST_AUTO_TEST_CASE(par_negative) {
  std::vector<int> empty;

  auto ret =
      std::fill_n(std::execution::par_unseq, empty.begin(), -1, 42);
  BOOST_CHECK(ret == empty.begin());
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types::type) {
  test_fill_n<T>(std::execution::par_unseq, 1000);
}

BOOST_AUTO_TEST_SUITE_END()
