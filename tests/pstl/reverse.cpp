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

BOOST_FIXTURE_TEST_SUITE(pstl_reverse, enable_unified_shared_memory)


template<class T, class Policy>
void test_reverse(Policy&& pol, std::size_t problem_size) {
  std::vector<T> device_data(problem_size), host_data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    device_data[i] = T{i};
    host_data[i] = T{i};
  }

  std::reverse(pol, device_data.begin(), device_data.end());
  std::reverse(host_data.begin(), host_data.end());

  BOOST_CHECK(device_data == host_data);
}

using types = boost::mp11::mp_list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_reverse<T>(std::execution::par_unseq, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_reverse<T>(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types) {
  test_reverse<T>(std::execution::par_unseq, 1000);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_odd, T, types) {
  test_reverse<T>(std::execution::par_unseq, 1001);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_reverse<T>(std::execution::par, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_reverse<T>(std::execution::par, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types) {
  test_reverse<T>(std::execution::par, 1000);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_odd, T, types) {
  test_reverse<T>(std::execution::par, 1001);
}
BOOST_AUTO_TEST_SUITE_END()
