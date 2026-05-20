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
  sycl::queue q{sycl::property::queue::in_order{}};
  if (q.get_device().has(sycl::aspect::usm_host_allocations)) {
    auto source = sycl::malloc_host(sizeof(int), q);
    auto dest = malloc(sizeof(int));

    q.memcpy(dest, source, sizeof(int)).wait();

    sycl::free(source, q);
    free(dest);
  }
}

BOOST_AUTO_TEST_CASE(inorder_queue_d2h_h2d_ordering) {
  sycl::queue q{sycl::property::queue::in_order{}};

  constexpr std::size_t N = 1 << 20;

  uint32_t *a = sycl::malloc_device<uint32_t>(N, q);
  uint32_t *b = sycl::malloc_device<uint32_t>(N, q);
  BOOST_REQUIRE(a);
  BOOST_REQUIRE(b);

  std::vector<uint32_t> host(N);
  std::vector<uint32_t> check(N);

  for (int iter = 1; iter <= 5; ++iter) {
    q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      auto i = idx[0];
      a[i] = static_cast<std::uint32_t>(iter) ^ static_cast<std::uint32_t>(i * 2654435761u);
    });
    q.memcpy(host.data(), a, N * sizeof(uint32_t));        // device -> host
    q.memcpy(b, host.data(), N * sizeof(uint32_t));        // host -> device
    q.memcpy(check.data(), b, N * sizeof(uint32_t)).wait();// device -> host

    for (std::size_t i = 0; i < N; ++i) {
      auto expected = static_cast<uint32_t>(iter) ^ static_cast<uint32_t>(i * 2654435761u);
      BOOST_CHECK_EQUAL(check[i], expected);
      if (check[i] != expected) {
        break;
      }
    }
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BOOST_AUTO_TEST_CASE(inorder_queue_d2h_h2h_h2d_ordering) {
  sycl::queue q{sycl::property::queue::in_order{}};

  constexpr std::size_t N = 1 << 20;

  uint32_t *dev = sycl::malloc_device<uint32_t>(N, q);
  BOOST_REQUIRE(dev);

  std::vector<uint32_t> host_a(N);
  std::vector<uint32_t> host_b(N);
  std::vector<uint32_t> check(N);

  for (int iter = 1; iter <= 5; ++iter) {
    q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> idx) {
      auto i = idx[0];
      dev[i] = static_cast<uint32_t>(iter) ^ static_cast<uint32_t>(i * 2654435761u);
    });
    q.memcpy(host_a.data(), dev, N * sizeof(uint32_t));           // device -> host
    q.memcpy(host_b.data(), host_a.data(), N * sizeof(uint32_t)); // host -> host
    // There might a bug in current OpenCL driver in CI for host-host memcpy synchronization.
    if(q.get_device().get_backend() == sycl::backend::ocl)
      q.wait();
    q.memcpy(dev, host_b.data(), N * sizeof(uint32_t));           // host -> device
    q.memcpy(check.data(), dev, N * sizeof(uint32_t)).wait();     // device -> host

    for (std::size_t i = 0; i < N; ++i) {
      auto expected = static_cast<uint32_t>(iter) ^ static_cast<uint32_t>(i * 2654435761u);
      BOOST_CHECK_EQUAL(check[i], expected);
      if (check[i] != expected) {
        break;
      }
    }
  }

  sycl::free(dev, q);
}

BOOST_AUTO_TEST_CASE(two_queues) {
  sycl::queue q1{sycl::property_list{sycl::property::queue::in_order{}}};
  sycl::queue q2(q1.get_context(), q1.get_device(),
		 sycl::property_list{sycl::property::queue::in_order{}});
  std::size_t test_size = 128;
  int *dev_ptr1 = sycl::malloc_device<int>(test_size, q1);
  int *dev_ptr2 = sycl::malloc_device<int>(test_size, q1);

  q1.memset(dev_ptr1, 0, test_size * sizeof(int));
  q2.memset(dev_ptr2, 0, test_size * sizeof(int));

  const unsigned iterations = 32;
  for (unsigned i = 0; i < iterations; i++) {
    q1.parallel_for(sycl::range<1>{test_size},
                    [=](sycl::id<1> idx) { dev_ptr1[idx] += 1; });

    q2.parallel_for(sycl::range<1>{test_size},
                    [=](sycl::id<1> idx) { dev_ptr2[idx] += 1; });
  }
  q1.wait();
  q2.wait();

  std::vector<int> host_ptr1(test_size);
  std::vector<int> host_ptr2(test_size);
  q1.memcpy(host_ptr1.data(), dev_ptr1, test_size * sizeof(int)).wait();
  q1.memcpy(host_ptr2.data(), dev_ptr2, test_size * sizeof(int));
  q1.wait();

  for (int i = 0; i < test_size; ++i) {
    BOOST_TEST(host_ptr1[i] == iterations);
    BOOST_TEST(host_ptr2[i] == iterations);
  }

  sycl::free(dev_ptr1, q1);
  sycl::free(dev_ptr2, q2);
}

BOOST_AUTO_TEST_SUITE_END()
