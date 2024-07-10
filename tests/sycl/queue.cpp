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

#include <numeric>
#include <type_traits>

#include "sycl_test_suite.hpp"
using namespace cl;

BOOST_FIXTURE_TEST_SUITE(queue_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(queue_wait) {
  sycl::queue q1;
  sycl::queue q2;

  auto evt1 = q1.single_task([=](){});
  auto evt2 = q2.single_task([=](){});

  BOOST_CHECK(q1.get_info<sycl::info::queue::AdaptiveCpp_node_group>() !=
              q2.get_info<sycl::info::queue::AdaptiveCpp_node_group>());

  q1.wait();
  BOOST_CHECK(evt1.get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete);
  q2.wait();
  BOOST_CHECK(evt2.get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete);
}

BOOST_AUTO_TEST_CASE(queue_memcpy_host_to_host) {
  try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};

    auto source = sycl::malloc_host(sizeof(int), q);
    auto dest = malloc(sizeof(int));

    q.memcpy(dest, source, sizeof(int)).wait();

    sycl::free(source, q);
    free(dest);
  } catch (sycl::exception e) {
    BOOST_CHECK(true); // Skip the test if no GPU available
  }
}

BOOST_AUTO_TEST_SUITE_END()
