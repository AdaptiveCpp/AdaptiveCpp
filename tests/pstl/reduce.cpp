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

#include <boost/test/tools/old/interface.hpp>
#include <numeric>
#include <execution>
#include <utility>
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_reduce, enable_unified_shared_memory)

template<class T>
void test_basic_reduction(T init, std::size_t size) {
  std::vector<T> data(size);
  for(std::size_t i = 0; i < data.size(); ++i)
    data[i] = static_cast<T>(i);

  T reference_result = std::reduce(
      data.begin(), data.end(), init, std::plus<>{});
  T res = std::reduce(std::execution::par_unseq,
      data.begin(), data.end(), init, std::plus<>{});
  BOOST_CHECK(res == reference_result);

  T res2 = std::reduce(std::execution::par_unseq,
      data.begin(), data.end(), init);
  BOOST_CHECK(res2 == res);

  T reference_result2 = std::reduce(
      data.begin(), data.end());
  T res3 = std::reduce(std::execution::par_unseq,
      data.begin(), data.end());
  BOOST_CHECK(reference_result2 == res3);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty_offset) {
  test_basic_reduction(10, 0);
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element_offset) {
  test_basic_reduction(10, 1);
}

BOOST_AUTO_TEST_CASE(par_unseq_incomplete_single_work_group_offset) {
  test_basic_reduction(10, 127);
}

BOOST_AUTO_TEST_CASE(par_unseq_medium_size) {
  test_basic_reduction(0, 1000);
}

BOOST_AUTO_TEST_CASE(par_unseq_large_size) {
  test_basic_reduction(0ll, 1000*1000);
}

BOOST_AUTO_TEST_SUITE_END()
