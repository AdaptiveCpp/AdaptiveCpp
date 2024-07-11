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

#include "hipSYCL/glue/kernel_launcher_data.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
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
      rt::kernel_launcher{glue::kernel_launcher_data{},
                          common::auto_small_vector<
                              std::unique_ptr<rt::backend_kernel_launcher>>{}},
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
