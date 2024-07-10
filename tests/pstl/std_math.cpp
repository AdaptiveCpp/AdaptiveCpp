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
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

namespace {

auto tolerance = boost::test_tools::tolerance(0.001f);

}

BOOST_FIXTURE_TEST_SUITE(pstl_std_math, enable_unified_shared_memory)

BOOST_AUTO_TEST_CASE(par_unseq) {
  std::vector<float> data(1000);
  for(int i = 0; i < data.size(); ++i)
    data[i] = static_cast<float>(i);
  
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), [](auto& x){
    x = std::sin(x) + std::pow(x, 0.01f);
  });

  for(int i = 0; i < data.size(); ++i) {
    float x = static_cast<float>(i);
    float reference_result = std::sin(x) + std::pow(x, 0.01f);

    BOOST_TEST(reference_result == data[i], tolerance);
  }
}

BOOST_AUTO_TEST_SUITE_END()
