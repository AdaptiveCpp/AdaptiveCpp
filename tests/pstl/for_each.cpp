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

BOOST_FIXTURE_TEST_SUITE(pstl_for_each, enable_unified_shared_memory)


BOOST_AUTO_TEST_CASE(par_unseq_zero_size) {
  std::vector<int> data{24};
  std::for_each(std::execution::par_unseq, data.begin(), data.begin(),
                [=](auto &x) { x = 12; });
  BOOST_CHECK(data[0] == 24);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_work_group) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;
  std::for_each(std::execution::par_unseq, data.begin(), data.end(),
                [=](auto &x) { x *= 2; });
  for(int i = 0; i < data.size(); ++i) {
    BOOST_CHECK(data[i] == 2*i);
  }
}


BOOST_AUTO_TEST_CASE(par_zero_size) {
  std::vector<int> data{24};
  std::for_each(std::execution::par, data.begin(), data.begin(),
                [=](auto &x) { x = 12; });
  BOOST_CHECK(data[0] == 24);
}

BOOST_AUTO_TEST_CASE(par_incomplete_work_group) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = i;
  std::for_each(std::execution::par, data.begin(), data.end(),
                [=](auto &x) { x *= 2; });
  for(int i = 0; i < data.size(); ++i) {
    BOOST_CHECK(data[i] == 2*i);
  }
}

BOOST_AUTO_TEST_SUITE_END()
