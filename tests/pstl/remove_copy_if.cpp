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
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_remove_copy_if, enable_unified_shared_memory)

template<class T, class Policy, class Generator, class UnaryPredicate>
void test_remove_copy_if(Policy&& pol, Generator&& gen, std::size_t problem_size,
                         UnaryPredicate p) {
  std::vector<T> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = gen(i);
  }

  std::vector<int> dest_device(problem_size);
  std::vector<int> dest_host(problem_size);

  auto ret = std::remove_copy_if(pol, data.begin(),
                              data.end(), dest_device.begin(), p);
  auto ret_reference = std::remove_copy_if(data.begin(), data.end(),
                                        dest_host.begin(), p);

  BOOST_CHECK(std::distance(dest_device.begin(), ret) ==
              std::distance(dest_host.begin(), ret_reference));

  BOOST_CHECK(dest_device == dest_host);
}

using types = boost::mp11::mp_list<int>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq,[](int i){return i+1;},
                         0, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element_match, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq, [](int i){return 42;},
                         1, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq, [](int i){return i+1;},
                         1, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_none, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq,[](int i){return i+1;},
                         1000, [](T x){return x == 4020;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_all, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq, [](int i){return 42;}, 
                         1000, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size_half, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq, 
                         [](int i){return (i%2==0 ? 42: i);}, 
                         1000, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_empty, T, types) {
  test_remove_copy_if<T>(std::execution::par,[](int i){return i+1;},
                         0, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element_match, T, types) {
  test_remove_copy_if<T>(std::execution::par_unseq, [](int i){return 42;},
                         1, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_single_element, T, types) {
  test_remove_copy_if<T>(std::execution::par, [](int i){return i+1;},
                         1, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_none, T, types) {
  test_remove_copy_if<T>(std::execution::par,[](int i){return i+1;},
                         1000, [](T x){return x == 4020;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_all, T, types) {
  test_remove_copy_if<T>(std::execution::par, [](int i){return 42;},
                         1000, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_medium_size_half, T, types) {
  test_remove_copy_if<T>(std::execution::par,
                         [](int i){return (i%2==0 ? 42: i);},
                         1000, [](T x){return x == 42;});
}

BOOST_AUTO_TEST_SUITE_END()
