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

BOOST_FIXTURE_TEST_SUITE(pstl_find_first_of, enable_unified_shared_memory)


template<class T, class Policy, class Generator>
void test_find_first_of(Policy&& pol, Generator&& gen, std::size_t problem_size) {
  std::vector<T> data(problem_size), s_data(problem_size);
  for(std::size_t i = 0; i < data.size(); ++i) {
    data[i] = gen(i);
    s_data[i] = (data[i]%3 == 0 ? gen(i) : gen(i)+40);
  }

  auto reference_result = std::find_first_of(data.begin(), data.end(), s_data.begin(),
                                             s_data.end());
  auto res = std::find_first_of(pol, data.begin(), data.end(), s_data.begin(),
                                s_data.end());
  BOOST_CHECK(res == reference_result);

  auto p = [](auto x, auto y) { return (x == y); };

  reference_result = std::find_first_of(data.begin(), data.end(), s_data.begin(),
                                        s_data.end(), p);
  res = std::find_first_of(pol, data.begin(), data.end(), s_data.begin(),
                                        s_data.end(), p);
  BOOST_CHECK(res == reference_result);

}

using types = boost::mp11::mp_list<int>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_find_first_of<T>(std::execution::par_unseq, [](int i){return i;}, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element_match, T, types) {
  test_find_first_of<T>(std::execution::par_unseq, [](int i){return 3;}, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_find_first_of<T>(std::execution::par_unseq, [](int i){return i+1;}, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size__multiple, T, types) {
  test_find_first_of<T>(std::execution::par_unseq, [](int i){return i+1;}, 1000);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types) {
  test_find_first_of<T>(std::execution::par_unseq, [](int i){return 42;}, 1000);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_find_first_of<T>(std::execution::par, [](int i){return i;}, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element_match, T, types) {
  test_find_first_of<T>(std::execution::par, [](int i){return 3;}, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_find_first_of<T>(std::execution::par, [](int i){return i+1;}, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_multiple, T, types) {
  test_find_first_of<T>(std::execution::par, [](int i){return i+1;}, 1000);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types) {
  test_find_first_of<T>(std::execution::par, [](int i){return 42;}, 1000);
}

BOOST_AUTO_TEST_SUITE_END()
