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

BOOST_FIXTURE_TEST_SUITE(pstl_unique, enable_unified_shared_memory)

template<class T, class Policy, class Generator, class BinaryPredicate = std::equal_to<>>
void test_unique(Policy&& pol, Generator&& gen, std::size_t problem_size,
                      BinaryPredicate pred = {}) {
  std::vector<T> dest_device(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    dest_device[i] = gen(i);
  }
  std::vector<T> dest_host = dest_device;

  auto ret = std::unique(pol, dest_device.begin(), dest_device.end(), pred);
  auto ret_reference = std::unique(dest_host.begin(), dest_host.end(), pred);

  BOOST_CHECK(std::distance(dest_device.begin(), ret) ==
              std::distance(dest_host.begin(), ret_reference));

  BOOST_CHECK(dest_device == dest_host);
}

using types = boost::mp11::mp_list<int>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_unique<T>(std::execution::par_unseq, [](int i){return i+1;}, T{0});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element_match, T, types) {
  test_unique<T>(std::execution::par_unseq, [](int i){return 42;}, T{1});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_unique<T>(std::execution::par_unseq, [](int i){return i+1;}, T{1});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_none, T, types) {
  test_unique<T>(std::execution::par_unseq, [](int i){return i+1;}, T{20});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_all, T, types) {
  test_unique<T>(std::execution::par_unseq, [](int i){return 42;}, T{20});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_half, T, types) {
  test_unique<T>(std::execution::par_unseq,
                      [](int i){return (i%5 != 0 ? 42: i);}, T{20});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_unique<T>(std::execution::par, [](int i){return i+1;}, T{0});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element_match, T, types) {
  test_unique<T>(std::execution::par, [](int i){return 42;}, T{1});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_unique<T>(std::execution::par, [](int i){return i+1;}, T{1});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_none, T, types) {
  test_unique<T>(std::execution::par, [](int i){return i+1;}, T{20});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_all, T, types) {
  test_unique<T>(std::execution::par, [](int i){return 42;}, T{20});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_half, T, types) {
  test_unique<T>(std::execution::par,
                      [](int i){return (i%5 != 0 ? 42: i);}, T{20});
}

BOOST_AUTO_TEST_SUITE_END()