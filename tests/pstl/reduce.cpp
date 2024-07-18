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

#include <boost/test/tools/old/interface.hpp>
#include <numeric>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_reduce, enable_unified_shared_memory)

template<class T, class Policy>
void test_basic_reduction(Policy&& pol, T init, std::size_t size) {
  std::vector<T> data(size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<T>(i);

  T reference_result = std::reduce(
      data.begin(), data.end(), init, std::plus<>{});
  T res = std::reduce(pol,
      data.begin(), data.end(), init, std::plus<>{});
  BOOST_CHECK(res == reference_result);

  T res2 = std::reduce(pol,
      data.begin(), data.end(), init);
  BOOST_CHECK(res2 == res);

  T reference_result2 = std::reduce(
      data.begin(), data.end());
  T res3 = std::reduce(pol, data.begin(), data.end());
  BOOST_CHECK(reference_result2 == res3);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty_offset) {
  test_basic_reduction(std::execution::par_unseq, 10, 0);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element_offset) {
  test_basic_reduction(std::execution::par_unseq, 10, 1);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_single_work_group_offset) {
  test_basic_reduction(std::execution::par_unseq, 10, 127);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_basic_reduction(std::execution::par_unseq, 0, 1000);
}

BOOST_AUTO_TEST_CASE(par_unseq_large_size) {
  test_basic_reduction(std::execution::par_unseq, 0ll, 1000*1000);
}



BOOST_AUTO_TEST_CASE(par_empty_offset) {
  test_basic_reduction(std::execution::par, 10, 0);
}

BOOST_AUTO_TEST_CASE(par_single_element_offset) {
  test_basic_reduction(std::execution::par, 10, 1);
}

BOOST_AUTO_TEST_CASE(par_incomplete_single_work_group_offset) {
  test_basic_reduction(std::execution::par, 10, 127);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  test_basic_reduction(std::execution::par, 0, 1000);
}

BOOST_AUTO_TEST_CASE(par_large_size) {
  test_basic_reduction(std::execution::par, 0ll, 1000*1000);
}


BOOST_AUTO_TEST_SUITE_END()
