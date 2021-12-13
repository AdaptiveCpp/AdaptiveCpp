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

  BOOST_CHECK(q1.get_info<sycl::info::queue::hipSYCL_node_group>() !=
              q2.get_info<sycl::info::queue::hipSYCL_node_group>());

  q1.wait();
  BOOST_CHECK(evt1.get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete);
  q2.wait();
  BOOST_CHECK(evt2.get_info<sycl::info::event::command_execution_status>() ==
              sycl::info::event_command_status::complete);
}

BOOST_AUTO_TEST_SUITE_END()
