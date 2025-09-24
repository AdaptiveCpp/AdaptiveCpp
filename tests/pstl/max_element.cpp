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

// #include <boost/test/tools/old/interface.hpp>
#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_max_element, enable_unified_shared_memory)

template<class Policy, class Generator, class Compare = std::less<>>
void test_max_element(Policy&& pol, std::size_t size, Generator&& gen,
                      Compare comp = {}) {
  std::vector<int> data(size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = gen(i);

  auto reference_result = std::max_element(data.begin(), data.end());
  auto res = std::max_element(pol, data.begin(), data.end());

  BOOST_CHECK(res == reference_result);

  reference_result = std::max_element(data.begin(), data.end(), comp);
  res = std::max_element(pol, data.begin(), data.end(), comp);

  BOOST_CHECK(res == reference_result);

  auto cmp = std::less_equal<>{};
  reference_result = std::max_element(data.begin(), data.end(), cmp);
  res = std::max_element(pol, data.begin(), data.end(), cmp);

  BOOST_CHECK(res == reference_result);
}

template<class Policy>
void empty_tests(Policy&& pol) {
  test_max_element(pol, 0, [](int i){return i;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_max_element(pol, 1, [](int i){return i + 1;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_max_element(pol, 200, [](int i){return -i - 1;});
  test_max_element(pol, 200, [](int i){return -200 + i;});
  test_max_element(pol, 200, [](int i){return 200-i;});
  test_max_element(pol, 200, [](int i){return (i > 10 ? i : 200-i);});
  test_max_element(pol, 200, [](int i){return (i <= 10 ? i : 200-i);});
  test_max_element(pol, 200, [](int i){return i + 1;});
  test_max_element(pol, 200, [](int i){return 42;});
  test_max_element(pol, 200, [](int i){return (i < 10 ?  42 : i);});
  test_max_element(pol, 200, [](int i){return (i < 10 ?  i : 42);});
  test_max_element(pol, 4, [](int i){return (i == 1 ? 42 : 100);});
  test_max_element(pol, 4, [](int i){return (i == 2 ? 42 : 100);});
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
