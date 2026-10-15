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

  BOOST_CHECK(
      device.has(sycl::aspect::usm_atomic_host_allocations) ==
      contains(aspects, sycl::aspect::usm_atomic_host_allocations));
  if (device.has(sycl::aspect::usm_atomic_host_allocations))
    BOOST_CHECK(device.has(sycl::aspect::usm_host_allocations));

  // For a one-device context, the context intersection must equal the device.
  const sycl::context context{device};
  BOOST_CHECK(
      context.get_info<
          sycl::info::context::atomic_memory_order_capabilities>() == orders);
  BOOST_CHECK(
      context.get_info<
          sycl::info::context::atomic_memory_scope_capabilities>() == scopes);
}

template <class T>
void check_system_acq_rel(sycl::queue& q, bool host) {
  T* data = host ? sycl::malloc_host<T>(2, q)
                 : sycl::malloc_shared<T>(2, q);
  BOOST_REQUIRE(data != nullptr);

  data[0] = 4;
  data[1] = 0;

  q.single_task([=]() {
     sycl::atomic_ref<T,
                      sycl::memory_order::acq_rel,
                      sycl::memory_scope::system,
                      sycl::access::address_space::global_space>
         atomic{data[0]};

     data[1] = atomic.fetch_add(T{3});
     T expected = 7;
     atomic.compare_exchange_strong(expected, T{8});
     atomic.fetch_or(T{3});
     atomic.fetch_and(T{7});
     atomic.fetch_xor(T{1});
   }).wait_and_throw();

  BOOST_CHECK(data[0] == T{2});
  BOOST_CHECK(data[1] == T{4});

  sycl::free(data, q);
}

BOOST_AUTO_TEST_CASE(reported_system_acq_rel_executes) {
  sycl::queue q;
  const sycl::device device = q.get_device();

  const auto orders = device.get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  const auto scopes = device.get_info<
      sycl::info::device::atomic_memory_scope_capabilities>();

  if (!contains(orders, sycl::memory_order::acq_rel) ||
      !contains(scopes, sycl::memory_scope::system)) {
    BOOST_TEST_MESSAGE("Skipping: device lacks acq_rel + system scope");
    return;
  }

  for (bool host : {true, false}) {
    const bool supported = host
        ? device.has(sycl::aspect::usm_atomic_host_allocations)
        : device.has(sycl::aspect::usm_atomic_shared_allocations);
    if (!supported) {
      BOOST_TEST_MESSAGE(host ? "Skipping atomic host USM"
                              : "Skipping atomic shared USM");
      continue;
    }
    check_system_acq_rel<std::uint32_t>(q, host);
    if (device.has(sycl::aspect::atomic64))
      check_system_acq_rel<std::uint64_t>(q, host);
  }
}

template <class T>
void check_floating_fetch_min_max(sycl::queue& q, sycl::memory_order order) {
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

       acc[1] = atomic.fetch_min(T{3}, order); // returns 4, stores 3
       acc[2] = atomic.fetch_max(T{7}, order); // returns 3, stores 7
       atomic.fetch_min(T{9}, order); // remains 7; still an RMW
       atomic.fetch_max(T{1}, order); // remains 7; still an RMW
     });
   }).wait_and_throw();

  sycl::host_accessor result{buffer};

  BOOST_CHECK(result[0] == T{7});
  BOOST_CHECK(result[1] == T{4});
  BOOST_CHECK(result[2] == T{3});
}

BOOST_AUTO_TEST_CASE(floating_fetch_min_max_return_old_value) {
  sycl::queue q;
  SKIP_IF_MOLTENVK(q.get_device())

  const auto orders = q.get_device().get_info<
      sycl::info::device::atomic_memory_order_capabilities>();
  for (auto order : {sycl::memory_order::relaxed, sycl::memory_order::release,
                     sycl::memory_order::acq_rel}) {
    if (!contains(orders, order))
      continue;
    check_floating_fetch_min_max<float>(q, order);

    if (q.get_device().has(sycl::aspect::fp64) &&
        q.get_device().has(sycl::aspect::atomic64)) {
      check_floating_fetch_min_max<double>(q, order);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
