/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#include <cstdint>

#include "../sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(smoke_task_graph_tests, reset_device_fixture)


BOOST_AUTO_TEST_CASE(task_graph_synchronization) {
  using namespace cl::sycl::access;
  constexpr size_t num_elements = 4096 * 1024;

  cl::sycl::queue q1;
  cl::sycl::queue q2;
  cl::sycl::queue q3;

  cl::sycl::buffer<int, 1> buf_a{num_elements};
  cl::sycl::buffer<int, 1> buf_b{num_elements};
  cl::sycl::buffer<int, 1> buf_c{num_elements};

  q1.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_init_a>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_a[tid] = static_cast<int>(tid.get(0));
      });
  });

  q2.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::read>(cgh);
    auto acc_b = buf_b.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_init_b>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_b[tid] = acc_a[tid];
      });
  });

  q3.submit([&](cl::sycl::handler& cgh) {
    auto acc_a = buf_a.get_access<mode::read>(cgh);
    auto acc_b = buf_b.get_access<mode::read>(cgh);
    auto acc_c = buf_c.get_access<mode::discard_write>(cgh);
    cgh.parallel_for<class tdag_sync_add_a_b>(cl::sycl::range<1>{num_elements},
      [=](cl::sycl::id<1> tid) {
        acc_c[tid] = acc_a[tid] + acc_b[tid];
      });
  });

  auto result = buf_c.get_access<mode::read>();
  for(size_t i = num_elements; i < num_elements; ++i) {
    BOOST_REQUIRE(result[i] == 2 * i);
  }
}


BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
