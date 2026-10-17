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

BOOST_FIXTURE_TEST_SUITE(atomic_host_rmw_regressions, reset_device_fixture)

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
  // This PR repairs the host implementations. GPU lowering is addressed by
  // subsequent PRs in the series; do not test it before those fixes exist.
  if(q.get_device().get_backend() != sycl::backend::omp) 
  {
    BOOST_TEST_MESSAGE("Skipping host atomic regression on non-CPU device");
    return;
  }

  for(auto order : {sycl::memory_order::relaxed, sycl::memory_order::release,
                    sycl::memory_order::acq_rel}) {
    check_floating_fetch_min_max<float>(q, order);

    if (q.get_device().has(sycl::aspect::fp64) &&
        q.get_device().has(sycl::aspect::atomic64)) {
      check_floating_fetch_min_max<double>(q, order);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
