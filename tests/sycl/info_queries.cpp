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
