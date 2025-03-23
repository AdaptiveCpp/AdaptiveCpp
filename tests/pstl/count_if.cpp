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

template<class T, class Policy, class UnaryPredicate>
void test_count_if(Policy&& pol, std::size_t problem_size, UnaryPredicate p) {
  std::vector<T> data(problem_size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<T>(i);


  auto reference_result = std::count_if(data.begin(), data.end(), p);
  auto res = std::count_if(pol, data.begin(), data.end(), p);
  
  BOOST_CHECK(res == reference_result);

  if (problem_size == 1000)
    BOOST_CHECK(res == 15);
  else if (problem_size == 1)
    BOOST_CHECK(res == 1);
  else if (problem_size == 0)
    BOOST_CHECK(res == 0);
}

using types = boost::mp11::mp_list<int>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_count_if<T>(std::execution::par_unseq, 0, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_count_if<T>(std::execution::par_unseq, 1, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types) {
  test_count_if<T>(std::execution::par_unseq, 1000, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_count_if<T>(std::execution::par, 0, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_count_if<T>(std::execution::par, 1, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types) {
  test_count_if<T>(std::execution::par, 1000, [](auto x) { return x < T{15}; });
}

BOOST_AUTO_TEST_SUITE_END()
