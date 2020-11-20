/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include <numeric>

#include "hipSYCL/sycl/libkernel/reduction.hpp"
#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(reduction_tests, reset_device_fixture)

template <class T, class Combiner>
void run_scalar_reduction_test(const Combiner &combiner, const T &identity,
                               const std::size_t test_size = 4096){
  
  std::vector<T> input_data(test_size);
  for(std::size_t i = 0; i < test_size; ++i)
    input_data[i] = static_cast<T>(i);

  T expected =
      std::accumulate(input_data.begin(), input_data.end(), identity, combiner);

  sycl::buffer<T> data_buff{input_data.data(), sycl::range<1>{test_size}};
  sycl::buffer<T> output_buff{sycl::range<1>{1}};

  sycl::queue q;
  q.submit([&](sycl::handler &cgh) {
    auto input_acc = data_buff.template get_access<sycl::access::mode::read>(cgh);
    auto output_acc =
        output_buff.template get_access<sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for<class reduction_kernel>(
        sycl::range<1>{test_size},
        sycl::reduction(output_acc, 0, sycl::plus<T>{}),
        [=](sycl::id<1> idx, auto &reducer) {
          reducer.combine(input_acc[idx]);
        });
  });
  {
    auto host_acc = output_buff.template get_access<sycl::access::mode::read>();

    auto result = host_acc[0];
    BOOST_TEST(result == expected);
  }
}

BOOST_AUTO_TEST_CASE(scalar_reduction) {

  run_scalar_reduction_test(sycl::plus<int>(), 0);
}

BOOST_AUTO_TEST_SUITE_END()