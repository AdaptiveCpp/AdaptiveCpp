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

BOOST_FIXTURE_TEST_SUITE(pstl_transform, enable_unified_shared_memory)

void run_unary_transform_test(std::size_t problem_size) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  std::vector<int> device_out(problem_size);
  std::vector<int> host_out(problem_size);

  auto transformation = [](auto x) {
    return x + 1;
  };

  auto ret = std::transform(std::execution::par_unseq, data.begin(), data.end(),
                            device_out.begin(), transformation);
  auto host_ret = std::transform(data.begin(), data.end(), host_out.begin(),
                                 transformation);

  BOOST_CHECK(device_out == host_out);
  BOOST_CHECK(ret == device_out.begin() + problem_size);
}


BOOST_AUTO_TEST_CASE(par_unseq_unary_zero_size) {
  run_unary_transform_test(0);
}

BOOST_AUTO_TEST_CASE(par_unseq_unary_size_1) {
  run_unary_transform_test(1);
}

BOOST_AUTO_TEST_CASE(par_unseq_unary_large) {
  run_unary_transform_test(1000*1000);
}


void run_binary_transform_test(std::size_t problem_size) {
  std::vector<int> data1(problem_size);
  for(int i = 0; i < data1.size(); ++i) {
    data1[i] = i;
  }

  std::vector<int> data2(problem_size);
  for(int i = 0; i < data2.size(); ++i) {
    data2[i] = -i+10;
  }


  std::vector<int> device_out(problem_size);
  std::vector<int> host_out(problem_size);

  auto transformation = [](auto x, auto y) {
    return x * y + y;
  };

  auto ret =
      std::transform(std::execution::par_unseq, data1.begin(), data1.end(),
                     data2.begin(), device_out.begin(), transformation);
  auto host_ret = std::transform(data1.begin(), data1.end(), data2.begin(),
                                 host_out.begin(), transformation);

  BOOST_CHECK(device_out == host_out);
  BOOST_CHECK(ret == device_out.begin() + problem_size);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_zero_size) {
  run_binary_transform_test(0);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_size_1) {
  run_binary_transform_test(1);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_large) {
  run_binary_transform_test(1000);
}


BOOST_AUTO_TEST_SUITE_END()
