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

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

BOOST_FIXTURE_TEST_SUITE(atomic_capability_tests, reset_device_fixture)

template <class T>
bool contains(const std::vector<T>& values, T value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

BOOST_AUTO_TEST_CASE(capability_queries) {
  sycl::queue q;
  const sycl::device device = q.get_device();

  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  BOOST_CHECK(contains(orders, sycl::memory_order::relaxed));
  BOOST_CHECK(contains(scopes, sycl::memory_scope::sub_group));
  BOOST_CHECK(contains(scopes, sycl::memory_scope::work_group));

  if (contains(orders, sycl::memory_order::acq_rel)) {
    BOOST_CHECK(contains(orders, sycl::memory_order::acquire));
    BOOST_CHECK(contains(orders, sycl::memory_order::release));
  }

  if (contains(scopes, sycl::memory_scope::system))
    BOOST_CHECK(contains(scopes, sycl::memory_scope::device));

  const auto aspects =
      device.get_info<sycl::info::device::aspects>();
  const bool atomic_shared_in_list =
      contains(aspects, sycl::aspect::usm_atomic_shared_allocations);

  BOOST_CHECK(
      device.has(sycl::aspect::usm_atomic_shared_allocations) ==
      atomic_shared_in_list);

  if (device.has(sycl::aspect::usm_atomic_shared_allocations))
    BOOST_CHECK(device.has(sycl::aspect::usm_shared_allocations));

  // For a one-device context, the context intersection must equal the device.
  const sycl::context context{device};
  BOOST_CHECK(
      context.get_info<
          sycl::info::context::atomic_memory_order_capabilities>() == orders);
  BOOST_CHECK(
      context.get_info<
          sycl::info::context::atomic_memory_scope_capabilities>() == scopes);
}

BOOST_AUTO_TEST_CASE(reported_system_acq_rel_executes) {
  sycl::queue q;
  const sycl::device device = q.get_device();

  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  if (!device.has(sycl::aspect::usm_shared_allocations) ||
      !device.has(sycl::aspect::usm_atomic_shared_allocations) ||
      !contains(orders, sycl::memory_order::acq_rel) ||
      !contains(scopes, sycl::memory_scope::system)) {
    BOOST_TEST_MESSAGE(
        "Skipping: device does not report atomic shared USM with "
        "acq_rel + system scope");
    return;
  }

  std::uint32_t* data = sycl::malloc_shared<std::uint32_t>(2, q);
  BOOST_REQUIRE(data != nullptr);

  data[0] = 4;
  data[1] = 0;

  q.single_task([=]() {
     sycl::atomic_ref<std::uint32_t,
                      sycl::memory_order::acq_rel,
                      sycl::memory_scope::system,
                      sycl::access::address_space::global_space>
         atomic{data[0]};

     data[1] = atomic.fetch_add(3);
   }).wait();

  BOOST_CHECK(data[0] == 7);
  BOOST_CHECK(data[1] == 4);

  sycl::free(data, q);
}

template <class T>
void check_floating_fetch_min_max(sycl::queue& q) {
  std::array<T, 3> data{T{4}, T{0}, T{0}};
  sycl::buffer<T> buffer{data.data(), sycl::range<1>{data.size()}};

  q.submit([&](sycl::handler& cgh) {
     sycl::accessor acc{buffer, cgh, sycl::read_write};

     cgh.single_task([=]() {
       sycl::atomic_ref<T,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space>
           atomic{acc[0]};

       acc[1] = atomic.fetch_min(T{3}); // returns 4, stores 3
       acc[2] = atomic.fetch_max(T{7}); // returns 3, stores 7
     });
   }).wait();

  sycl::host_accessor result{buffer};

  BOOST_CHECK(result[0] == T{7});
  BOOST_CHECK(result[1] == T{4});
  BOOST_CHECK(result[2] == T{3});
}

BOOST_AUTO_TEST_CASE(floating_fetch_min_max_return_old_value) {
  sycl::queue q;
  SKIP_IF_MOLTENVK(q.get_device())

  check_floating_fetch_min_max<float>(q);

  if (q.get_device().has(sycl::aspect::fp64) &&
      q.get_device().has(sycl::aspect::atomic64)) {
    check_floating_fetch_min_max<double>(q);
  }
}

BOOST_AUTO_TEST_SUITE_END()
