/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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
#include <iostream>

#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(info_queries, reset_device_fixture)

BOOST_AUTO_TEST_CASE(device_queries) {
  
  sycl::device d{sycl::default_selector{}};
  sycl::queue q{d};

  std::string device = d.get_info<sycl::info::device::name>();
  BOOST_TEST(device.length() > 0);
  std::cout << "Default-selected queue runs on device: " << device << std::endl;
  
  // TODO Add tests for more queries
}
BOOST_AUTO_TEST_CASE(kernel_specific_queries) {
  
  sycl::queue q;

  sycl::program p{q.get_context()};
  sycl::kernel k = p.get_kernel<void>();

  auto wg_size =
      k.get_info<sycl::info::kernel_device_specific::work_group_size>(
          q.get_device());
  BOOST_TEST(wg_size > 0);

  auto max_sgs =
      k.get_info<sycl::info::kernel_device_specific::max_num_sub_groups>(
          q.get_device());
  BOOST_TEST(max_sgs > 0);

  auto max_sg_size =
      k.get_info<sycl::info::kernel_device_specific::max_sub_group_size>(
          q.get_device());
  BOOST_TEST(max_sg_size > 0);
  
  // TODO Add tests for more queries
}

BOOST_AUTO_TEST_SUITE_END()
