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

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_ptr_validation, enable_unified_shared_memory)


template<class F>
void test_call(F f) {
  std::vector<int> data(1000);
  for(int i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  std::vector<int> output_pstl(data.size());
  std::vector<int> output_host(data.size());
  std::transform(std::execution::par_unseq, data.begin(), data.end(), output_pstl.begin(), f);

  for(int i = 0; i < data.size(); ++i) {
    output_host[i] = f(data[i]);
  }

  BOOST_CHECK(output_pstl == output_host);
}


BOOST_AUTO_TEST_CASE(unused_capture_by_ref) {
  test_call([&](int x){return x+1;});
}

BOOST_AUTO_TEST_CASE(used_capture_by_ref) {
  int c = 4;
  test_call([&](int x){return x+c;});
}


BOOST_AUTO_TEST_SUITE_END()
