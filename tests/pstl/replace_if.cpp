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

BOOST_FIXTURE_TEST_SUITE(pstl_replace_if, enable_unified_shared_memory)

template <class Policy, class Generator, class Pred>
void test_replace_if(Policy &&pol, std::size_t problem_size, Generator gen,
                     Pred p, int new_val) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i)
    data[i] = gen(i);
  std::vector<int> host_data = data;

  std::replace_if(pol, data.begin(), data.end(), p,
                  new_val);
  std::replace_if(host_data.begin(), host_data.end(), p, new_val);
  BOOST_CHECK(data == host_data);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_replace_if(std::execution::par_unseq,
      0, [](int i) { return i; }, [](int x) { return x % 2 == 0; }, 4);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_replace_if(std::execution::par_unseq,
      1, [](int i) { return 42; }, [](int x) { return x == 42; }, 4);
  test_replace_if(std::execution::par_unseq,
      1, [](int i) { return 2; }, [](int x) { return x == 42; }, 4);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_replace_if(std::execution::par_unseq,
      1000, [](int i) { return i%10+3; }, [](int x) { return x == 20; }, 4);
  test_replace_if(std::execution::par_unseq,
      1000, [](int i) { return i%10+3; }, [](int x) { return x == -2; }, 4);
}


BOOST_AUTO_TEST_CASE(par_empty) {
  test_replace_if(std::execution::par,
      0, [](int i) { return i; }, [](int x) { return x % 2 == 0; }, 4);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  test_replace_if(std::execution::par,
      1, [](int i) { return 42; }, [](int x) { return x == 42; }, 4);
  test_replace_if(std::execution::par,
      1, [](int i) { return 2; }, [](int x) { return x == 42; }, 4);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  test_replace_if(std::execution::par,
      1000, [](int i) { return i%10+3; }, [](int x) { return x == 20; }, 4);
  test_replace_if(std::execution::par,
      1000, [](int i) { return i%10+3; }, [](int x) { return x == -2; }, 4);
}


BOOST_AUTO_TEST_SUITE_END()
