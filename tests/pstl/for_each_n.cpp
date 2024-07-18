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
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_for_each_n, enable_unified_shared_memory)

template<class Policy>
void test_zero_size(Policy&& pol) {
  std::vector<int> data{24};
  auto res = std::for_each_n(pol, data.begin(), 0,
                             [=](auto &x) { x = 12; });
  BOOST_CHECK(data[0] == 24);
  BOOST_CHECK(res == data.begin());
}

template<class Policy>
void test_incomplete_work_group(Policy&& pol) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;
  auto res = std::for_each_n(pol, data.begin(),
                             data.size(), [=](auto &x) { x *= 2; });
  for(int i = 0; i < data.size(); ++i) {
    BOOST_CHECK(data[i] == 2*i);
  }
  BOOST_CHECK(res == data.end());
}

BOOST_AUTO_TEST_CASE(par_unseq_zero_size) {
  test_zero_size(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_work_group) {
  test_incomplete_work_group(std::execution::par_unseq);
}


BOOST_AUTO_TEST_CASE(par_zero_size) {
  test_zero_size(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_incomplete_work_group) {
  test_incomplete_work_group(std::execution::par);
}

BOOST_AUTO_TEST_SUITE_END()
