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


#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

BOOST_FIXTURE_TEST_SUITE(interop_handle_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(interop_handle_api) {
  namespace s = cl::sycl;
  using namespace hipsycl;

  rt::device_id assigned_device{rt::backend_descriptor{rt::hardware_platform::cpu,
                                rt::api_platform::omp}, 12345};

  rt::backend_executor *executor = nullptr;
  s::interop_handle ih{assigned_device, executor};
  s::backend b = ih.get_backend();

  BOOST_CHECK(b == s::backend::omp);
}

BOOST_AUTO_TEST_SUITE_END()
