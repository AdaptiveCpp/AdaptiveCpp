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

BOOST_FIXTURE_TEST_SUITE(pstl_copy_n, enable_unified_shared_memory)

template<class T, class Policy>
void test_copy_n(Policy&& pol, std::size_t problem_size) {
  std::vector<T> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = T{i};
  }

  std::vector<T> dest_device(problem_size);
  std::vector<T> dest_host(problem_size);

  auto ret = std::copy_n(pol, data.begin(), data.size(),
                         dest_device.begin());
  std::copy_n(data.begin(), data.size(), dest_host.begin());

  BOOST_CHECK(ret == dest_device.begin() + problem_size);
  BOOST_CHECK(dest_device == dest_host);
}

BOOST_AUTO_TEST_CASE(par_unseq_negative) {
  std::vector<int> empty;
  std::vector<int> dest(1);

  auto ret =
      std::copy_n(std::execution::par_unseq, empty.begin(), -1, dest.begin());
  BOOST_CHECK(ret == dest.begin());
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types::type) {
  test_copy_n<T>(std::execution::par_unseq, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types::type) {
  test_copy_n<T>(std::execution::par_unseq, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types::type) {
  test_copy_n<T>(std::execution::par_unseq, 1000);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types::type) {
  test_copy_n<T>(std::execution::par, 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types::type) {
  test_copy_n<T>(std::execution::par, 1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size, T, types::type) {
  test_copy_n<T>(std::execution::par, 1000);
}


BOOST_AUTO_TEST_SUITE_END()
