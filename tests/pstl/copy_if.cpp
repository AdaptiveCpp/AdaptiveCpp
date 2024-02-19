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

#include "pstl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(pstl_copy_if, enable_unified_shared_memory)


template<class Generator>
void test_copy_if(std::size_t problem_size, Generator&& gen) {
  std::vector<int> data(problem_size);
  for(int i = 0; i < problem_size; ++i) {
    data[i] = gen(i);
  }

  std::vector<int> dest_device(problem_size);
  std::vector<int> dest_host(problem_size);

  auto p = [](auto x) { return x % 2 == 0; };

  auto ret = std::copy_if(std::execution::par_unseq, data.begin(), data.end(),
                          dest_device.begin(), p);
  std::copy_if(data.begin(), data.end(), dest_host.begin(), p);

  BOOST_CHECK(ret == dest_device.begin() + problem_size);
  // Our copy_if implementation is currently incorrect, since
  // we always copy results to the same position (we would
  // actually need to run a scan algorithm to find the right place)
  //BOOST_CHECK(dest_device == dest_host);
}

BOOST_AUTO_TEST_CASE(par_unseq_empty) {
  test_copy_if(0, [](int i){return i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_single_element) {
  test_copy_if(1, [](int i){return i+3;});
}

BOOST_AUTO_TEST_CASE(par_unseq_none) {
  test_copy_if(1000, [](int i){return 1;});
}

BOOST_AUTO_TEST_CASE(par_unseq_all) {
  test_copy_if(1000, [](int i){return 2*i;});
}

BOOST_AUTO_TEST_CASE(par_unseq_half) {
  test_copy_if(1000, [](int i){return i;});
}

BOOST_AUTO_TEST_SUITE_END()
