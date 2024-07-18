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
#include <pstl/glue_execution_defs.h>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_any_of, enable_unified_shared_memory)

template <class Policy, class Generator, class Predicate>
void test_any_of(Policy&& pol, std::size_t problem_size, Generator gen, Predicate p) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);

  auto ret =
      std::any_of(pol, data.begin(), data.end(), p);
  auto ret_host =
      std::any_of(data.begin(), data.end(), p);

  BOOST_CHECK(ret == ret_host);
}

template<class Policy>
void empty_tests(Policy&& pol) {
  test_any_of(pol, 0, [](int i){return i;}, [](int x){ return x > 0;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_any_of(pol, 1, [](int i){return i;}, [](int x){ return x < 0;});
  test_any_of(pol, 1, [](int i){return i;}, [](int x){ return x > 0;});
  test_any_of(pol, 1, [](int i){return i;}, [](int x){ return x >= 0;});
  test_any_of(pol, 1, [](int i){return i;}, [](int x){ return x % 2 == 0;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_any_of(pol, 1000, [](int i){return i;}, [](int x){ return x < 0;});
  test_any_of(pol, 1000, [](int i){return i;}, [](int x){ return x > 0;});
  test_any_of(pol, 1000, [](int i){return i;}, [](int x){ return x >= 0;});
  test_any_of(pol, 1000, [](int i){return i;}, [](int x){ return x % 2 == 0;});
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
  empty_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  single_element_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  medium_size_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_SUITE_END()
