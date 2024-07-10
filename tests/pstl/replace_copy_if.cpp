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

BOOST_FIXTURE_TEST_SUITE(pstl_replace_copy_if, enable_unified_shared_memory)

template <class Generator, class Pred>
void test_replace_copy_if(std::size_t problem_size, Generator gen, Pred p, int new_val) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);

  std::vector<int> dest(problem_size);
  std::vector<int> host_dest(problem_size);

  auto ret = std::replace_copy_if(std::execution::par_unseq, data.begin(),
                                  data.end(), dest.begin(), p, new_val);
  std::replace_copy_if(data.begin(), data.end(), host_dest.begin(), p, new_val);
  BOOST_CHECK(dest == host_dest);
  BOOST_CHECK(ret == dest.begin() + problem_size);
}


BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_replace_copy_if(
      0, [](int i) { return i; }, [](int x) { return x % 2 == 0; }, 4);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_replace_copy_if(
      1, [](int i) { return 42; }, [](int x) { return x == 42; }, 4);
  test_replace_copy_if(
      1, [](int i) { return 2; }, [](int x) { return x == 42; }, 4);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_replace_copy_if(
      1000, [](int i) { return i%10+3; }, [](int x) { return x == 20; }, 4);
  test_replace_copy_if(
      1000, [](int i) { return i%10+3; }, [](int x) { return x == -2; }, 4);
}



BOOST_AUTO_TEST_SUITE_END()
