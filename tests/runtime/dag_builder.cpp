/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/runtime/application.hpp"
#include "runtime_test_suite.hpp"

#include <vector>
#include <memory>
#include <hipSYCL/runtime/dag_builder.hpp>

using namespace hipsycl;

BOOST_FIXTURE_TEST_SUITE(dag_builder, reset_device_fixture)
BOOST_AUTO_TEST_CASE(default_hints) {
  rt::runtime_keep_alive_token rt;
  rt::execution_hints hints;
  // Construct imaginary device
  rt::device_id id{rt::backend_descriptor{rt::hardware_platform::cpu,
                                          rt::api_platform::omp},
                   12345};
  
  hints.set_hint(rt::hints::bind_to_device{id});
  rt::dag_builder builder{rt.get()};

  auto reqs = rt::requirements_list{rt.get()};

  auto dummy_kernel_op = rt::make_operation<rt::kernel_operation>(
      "test_kernel",
      common::auto_small_vector<std::unique_ptr<rt::backend_kernel_launcher>>{},
      reqs);

  rt::dag_node_ptr node = builder.add_command_group(
      std::move(dummy_kernel_op), reqs, hints);

  rt::execution_hints& node_hints = node->get_execution_hints();
  BOOST_CHECK(node_hints.has_hint<rt::hints::bind_to_device>());
  BOOST_CHECK(
      node_hints.get_hint<rt::hints::bind_to_device>()->get_device_id() == id);

  node->cancel();
}

BOOST_AUTO_TEST_SUITE_END()
