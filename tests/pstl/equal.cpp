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
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_equal, enable_unified_shared_memory)

template <class Policy, class Generator>
void test_equal(Policy&& pol, std::size_t problem_size, Generator gen) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);
  
  std::vector<int> data2(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data2[i] = gen(i);

  auto ret =
      std::equal(pol, data.begin(), data.end(), data2.begin());
  auto ret_host =
      std::equal(data.begin(), data.end(), data2.begin());

  BOOST_CHECK(ret == ret_host);
  std::cout << "\n testing...\n";
  std::cout << ret << "\n" << ret_host;
  std::cout << "\n end.\n";
}


template<class Policy>
void empty_tests(Policy&& pol) {
  test_equal(pol, 0, [](int i){return i;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_equal(pol, 1, [](int i){return i;});
  test_equal(pol, 1, [](int i){return i;});
  test_equal(pol, 1, [](int i){return i;});
  test_equal(pol, 1, [](int i){return i;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_equal(pol, 1000, [](int i){return i;});
  test_equal(pol, 1000, [](int i){return i;});
  test_equal(pol, 1000, [](int i){return i;});
  test_equal(pol, 1000, [](int i){return i;});
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
