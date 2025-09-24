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

// #include <boost/test/tools/old/interface.hpp>
#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_mismatch, enable_unified_shared_memory)

template<class Policy, class Generator1, class Generator2, class BinaryPredicate = std::equal_to<>>
void test_mismatch(Policy&& pol, std::size_t size, Generator1&& gen1,
                      Generator2&& gen2, BinaryPredicate p = {}) {
  std::vector<int> data(size);
  std::vector<int> data_match(size);
  for(std::size_t i = 0; i < data.size(); ++i) {
    data[i] = gen1(i);
    data_match[i] = gen2(i);
  }

  auto res = std::mismatch(pol, data.begin(), data.end(),
                           data_match.begin());
  auto reference_result = std::mismatch(data.begin(), data.end(),
                                        data_match.begin());

  BOOST_CHECK(res == reference_result);

  res = std::mismatch(pol, data.begin(), data.end(),
                           data_match.begin(), p);
  reference_result = std::mismatch(data.begin(), data.end(),
                                        data_match.begin(), p);

  BOOST_CHECK(res == reference_result);

  std::vector<int> shorter_match(size);
  if (data.size()>1)
    shorter_match.resize(size/3);

  for(std::size_t i = 0; i < shorter_match.size(); ++i) {
    shorter_match[i] = gen2(i);
  }

  res = std::mismatch(pol, data.begin(), data.end(),
                      shorter_match.begin(), shorter_match.end());
  reference_result = std::mismatch(data.begin(), data.end(),
                            shorter_match.begin(), shorter_match.end());

  BOOST_CHECK(res == reference_result);

  res = std::mismatch(pol, data.begin(), data.end(),
                      shorter_match.begin(), shorter_match.end(), p);
  reference_result = std::mismatch(data.begin(), data.end(),
                            shorter_match.begin(), shorter_match.end(), p);

  BOOST_CHECK(res == reference_result);
}

template<class Policy>
void empty_tests(Policy&& pol) {
  test_mismatch(pol, 0, [](int i){return i;}, [](int i){return i;});
}

template<class Policy>
void single_element_tests(Policy&& pol) {
  test_mismatch(pol, 1, [](int i){return i + 1;}, [](int i){return i + 1;});
  test_mismatch(pol, 1, [](int i){return i + 1;}, [](int i){return i + 2;});
}

template<class Policy>
void medium_size_tests(Policy&& pol) {
  test_mismatch(pol, 20, [](int i){return -20 + i;}, [](int i){return -20 + i;});
  test_mismatch(pol, 20, [](int i){return (i == 0 ? 42 : i-1);}, [](int i){return (i == 0 ? 42 : i+1);});
  test_mismatch(pol, 20, [](int i){return (i > 10 ? 42 : 3);}, [](int i){return (i > 10 ? 42 : 4);});
  test_mismatch(pol, 20, [](int i){return (i < 10 ? 42 : 3);}, [](int i){return (i < 10 ? 42 : 4);});
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  empty_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  single_element_tests(std::execution::par_unseq);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  medium_size_tests(std::execution::par_unseq);
}


BOOST_AUTO_TEST_CASE(par_empty) {
  empty_tests(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_single_element) {
  single_element_tests(std::execution::par);
}

BOOST_AUTO_TEST_CASE(par_medium_size) {
  medium_size_tests(std::execution::par);
}


BOOST_AUTO_TEST_SUITE_END()