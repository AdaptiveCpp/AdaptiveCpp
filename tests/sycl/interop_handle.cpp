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
  namespace s = sycl;
  using namespace hipsycl;

  rt::device_id assigned_device{rt::backend_descriptor{rt::hardware_platform::cpu,
                                rt::api_platform::omp}, 0};
  s::queue q{s::device{assigned_device}};

  q.submit([&](s::handler &cgh) {
    cgh.AdaptiveCpp_enqueue_custom_operation([=](s::interop_handle &ih) {
      s::backend b = ih.get_backend();
      BOOST_CHECK(b == s::backend::omp);
    });
  });
  q.wait_and_throw();
}

#if defined(SYCL_EXT_ACPP_BACKEND_METAL) && \
    defined(ACPP_EXT_GET_NATIVE_ALLOCATION)

BOOST_AUTO_TEST_CASE(metal_get_native_allocation) {
  namespace s = sycl;

  s::queue q;
  if (q.get_device().get_backend() != s::backend::metal) {
    BOOST_TEST_MESSAGE("Skipping metal_get_native_allocation: not a Metal device");
    return;
  }

  constexpr std::size_t n = 64;
  int *ptr = s::malloc_shared<int>(n, q);
  BOOST_REQUIRE(ptr != nullptr);

  static_assert(
      s::AdaptiveCpp_can_get_native_allocation<s::backend::metal>);
  auto native_allocation =
      s::AdaptiveCpp_get_native_allocation<s::backend::metal>(
          ptr, q.get_context());

  BOOST_CHECK(native_allocation.buffer != nullptr);
  BOOST_CHECK_EQUAL(native_allocation.offset, 0u);

  s::free(ptr, q);
}

#endif // SYCL_EXT_ACPP_BACKEND_METAL && ACPP_EXT_GET_NATIVE_ALLOCATION

BOOST_AUTO_TEST_SUITE_END()
