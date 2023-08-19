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
#include <vector>

#include <boost/test/unit_test.hpp>

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_transform, enable_unified_shared_memory)

void run_unary_transform_test(std::size_t problem_size) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < data.size(); ++i) {
    data[i] = i;
  }

  std::vector<int> device_out(problem_size);
  std::vector<int> host_out(problem_size);

  auto transformation = [](auto x) {
    return x + 1;
  };

  auto ret = std::transform(std::execution::par_unseq, data.begin(), data.end(),
                            device_out.begin(), transformation);
  auto host_ret = std::transform(data.begin(), data.end(), host_out.begin(),
                                 transformation);

  BOOST_CHECK(device_out == host_out);
  BOOST_CHECK(ret == device_out.begin() + problem_size);
}


BOOST_AUTO_TEST_CASE(par_unseq_unary_zero_size) {
  run_unary_transform_test(0);
}

BOOST_AUTO_TEST_CASE(par_unseq_unary_size_1) {
  run_unary_transform_test(1);
}

BOOST_AUTO_TEST_CASE(par_unseq_unary_large) {
  run_unary_transform_test(1000*1000);
}


void run_binary_transform_test(std::size_t problem_size) {
  std::vector<int> data1(problem_size);
  for(int i = 0; i < data1.size(); ++i) {
    data1[i] = i;
  }

  std::vector<int> data2(problem_size);
  for(int i = 0; i < data2.size(); ++i) {
    data2[i] = -i+10;
  }


  std::vector<int> device_out(problem_size);
  std::vector<int> host_out(problem_size);

  auto transformation = [](auto x, auto y) {
    return x * y + y;
  };

  auto ret =
      std::transform(std::execution::par_unseq, data1.begin(), data1.end(),
                     data2.begin(), device_out.begin(), transformation);
  auto host_ret = std::transform(data1.begin(), data1.end(), data2.begin(),
                                 host_out.begin(), transformation);

  BOOST_CHECK(device_out == host_out);
  BOOST_CHECK(ret == device_out.begin() + problem_size);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_zero_size) {
  run_binary_transform_test(0);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_size_1) {
  run_binary_transform_test(1);
}

BOOST_AUTO_TEST_CASE(par_unseq_binary_large) {
  run_binary_transform_test(1000);
}


BOOST_AUTO_TEST_SUITE_END()
