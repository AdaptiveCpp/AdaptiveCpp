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

BOOST_FIXTURE_TEST_SUITE(pstl_none_of, enable_unified_shared_memory)

template <class Generator, class Predicate>
void test_none_of(std::size_t problem_size, Generator gen, Predicate p) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);

  auto ret =
      std::none_of(std::execution::par_unseq, data.begin(), data.end(), p);
  auto ret_host =
      std::none_of(data.begin(), data.end(), p);

  BOOST_CHECK(ret == ret_host);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_none_of(0, [](int i){return i;}, [](int x){ return x > 0;});
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_none_of(1, [](int i){return i;}, [](int x){ return x < 0;});
  test_none_of(1, [](int i){return i;}, [](int x){ return x > 0;});
  test_none_of(1, [](int i){return i;}, [](int x){ return x >= 0;});
  test_none_of(1, [](int i){return i;}, [](int x){ return x % 2 == 0;});
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_none_of(1000, [](int i){return i;}, [](int x){ return x < 0;});
  test_none_of(1000, [](int i){return i;}, [](int x){ return x > 0;});
  test_none_of(1000, [](int i){return i;}, [](int x){ return x >= 0;});
  test_none_of(1000, [](int i){return i;}, [](int x){ return x % 2 == 0;});
}



BOOST_AUTO_TEST_SUITE_END()
