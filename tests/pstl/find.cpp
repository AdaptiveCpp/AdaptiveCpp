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

BOOST_FIXTURE_TEST_SUITE(pstl_find, enable_unified_shared_memory)


template<class T, class Policy, class Generator>
void test_find(Policy&& pol, Generator&& gen, std::size_t problem_size, const T& value) {
  std::vector<T> data(problem_size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = gen(i);

  auto reference_result = std::find(data.begin(), data.end(), value);
  auto res = std::find(pol, data.begin(), data.end(), value);

  BOOST_CHECK(res == reference_result);
}

using types = boost::mp11::mp_list<int>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return i;}, 0, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element_match, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return 42;}, 1, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return i;}, 1, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_match, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return i;}, 1000, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_multiple, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return 42;}, 1000, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types) {
  test_find<T>(std::execution::par_unseq, [](int i){return i;}, 1000, T{4020});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_find<T>(std::execution::par, [](int i){return i;}, 0, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element_match, T, types) {
  test_find<T>(std::execution::par, [](int i){return 42;}, 1, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_find<T>(std::execution::par, [](int i){return i;}, 1, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_match, T, types) {
  test_find<T>(std::execution::par, [](int i){return i;}, 1000, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_multiple, T, types) {
  test_find<T>(std::execution::par, [](int i){return 42;}, 1000, T{42});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types) {
  test_find<T>(std::execution::par, [](int i){return i;}, 1000, T{4020});
}

BOOST_AUTO_TEST_SUITE_END()
