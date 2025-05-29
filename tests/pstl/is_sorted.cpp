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

BOOST_FIXTURE_TEST_SUITE(pstl_is_sorted, enable_unified_shared_memory)

template <class Policy, class Generator, class Compare = std::less<>>
void test_is_sorted(Policy&& pol, std::size_t problem_size,
                    Generator&& gen, Compare comp = {}) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);

  auto ret = std::is_sorted(pol, data.begin(), data.end());
  auto ret_host = std::is_sorted(data.begin(), data.end());

  BOOST_CHECK(ret == ret_host);

  ret = std::is_sorted(pol, data.begin(), data.end(), comp);
  ret_host = std::is_sorted(data.begin(), data.end(), comp);

  BOOST_CHECK(ret == ret_host);

  auto cmp = std::less_equal<>{};
  ret = std::is_sorted(pol, data.begin(), data.end(), cmp);
  ret_host = std::is_sorted(data.begin(), data.end(), cmp);

  BOOST_CHECK(ret == ret_host);

}


template<class Policy>
void empty_tests(Policy&& pol) {
  test_is_sorted(pol, 0, [](int i){return i;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_is_sorted(pol, 1, [](int i){return i;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_is_sorted(pol, 200, [](int i){return -i;});
  test_is_sorted(pol, 200, [](int i){return -200 + i;});
  test_is_sorted(pol, 200, [](int i){return 200-i;});
  test_is_sorted(pol, 200, [](int i){return (i > 50 ? i : 200-i);});
  test_is_sorted(pol, 200, [](int i){return (i <= 50 ? i : 200-i);});
  test_is_sorted(pol, 200, [](int i){return i;});
  test_is_sorted(pol, 200, [](int i){return 42;});
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


// TODO: investigate UB failures likely caused by the interplay
// between libstdc++, tbb, and boost; observed across all pstl
// ci runs.
// BOOST_AUTO_TEST_CASE(par_empty) {
//   empty_tests(std::execution::par);
// }

// BOOST_AUTO_TEST_CASE(par_single_element) {
//   single_element_tests(std::execution::par);
// }

// BOOST_AUTO_TEST_CASE(par_medium_size) {
//   medium_size_tests(std::execution::par);
// }


BOOST_AUTO_TEST_SUITE_END()
