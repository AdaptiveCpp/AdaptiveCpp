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
