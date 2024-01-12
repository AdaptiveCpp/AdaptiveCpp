/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_fill_n, enable_unified_shared_memory)


template<class T>
void test_fill_n(std::size_t problem_size) {
  std::vector<T> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = T{i};
  }

  auto ret =
      std::fill_n(std::execution::par_unseq, data.begin(), data.size(), T(42));

  BOOST_CHECK(ret == data.begin() + problem_size);

  std::vector<T> host_data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
   host_data[i] = T{i};
  }

  std::fill_n(host_data.begin(), host_data.size(), T(42));

  BOOST_CHECK(host_data == data);
}

BOOST_AUTO_TEST_CASE(par_unseq_negative) {
  std::vector<int> empty;

  auto ret =
      std::fill_n(std::execution::par_unseq, empty.begin(), -1, 42);
  BOOST_CHECK(ret == empty.begin());
}

using types = boost::mpl::list<int, non_trivial_copy>;
BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_empty, T, types::type) {
  test_fill_n<T>(0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_single_element, T, types::type) {
  test_fill_n<T>(1);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(par_unseq_medium_size, T, types::type) {
  test_fill_n<T>(1000);
}

BOOST_AUTO_TEST_SUITE_END()
