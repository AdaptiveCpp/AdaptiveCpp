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

#include <unordered_map>

BOOST_FIXTURE_TEST_SUITE(reference_semantics, reset_device_fixture)



BOOST_AUTO_TEST_CASE(hash) {
  // Currently only compile-testing
  std::unordered_map<cl::sycl::device, int> device_map;
  std::unordered_map<cl::sycl::context, int> context_map;
  std::unordered_map<cl::sycl::platform, int> platform_map;
  std::unordered_map<cl::sycl::buffer<int>, int> buffer_map;
}


BOOST_AUTO_TEST_SUITE_END()
