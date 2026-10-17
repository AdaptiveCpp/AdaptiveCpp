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
#include <array>
#include <cstddef>

BOOST_FIXTURE_TEST_SUITE(atomic_host_rmw_regressions, reset_device_fixture)

template <class T>
void check_floating_fetch_min_max(sycl::queue& queue, sycl::memory_order order) {

  //index
  static constexpr size_t  ValIdx = 0;
  static constexpr size_t  FetchMinStorageIdx = 1;
  static constexpr size_t  FetchMaxStorageIdx = 2;
  static constexpr size_t  SIZE_TEST = 3; // len

  //value
  static constexpr T InitialValue = static_cast<T>(4);
  static constexpr T MinValue = static_cast<T>(3);
  static constexpr T MaxValueLast = static_cast<T>(7);

  static constexpr T InvalidMinValue = static_cast<T>(MaxValueLast + 2);
  static constexpr T InvalidMaxValue = static_cast<T>(MaxValueLast - 2);

  std::array<T, SIZE_TEST> data{T{InitialValue}, T{0}, T{0}};
  sycl::buffer<T> buffer{data.data(), sycl::range<1>{data.size()}};

  queue.submit([&](sycl::handler& cgh) 
  {
     sycl::accessor acc{buffer, cgh, sycl::read_write};
     cgh.single_task([=]() 
      {
        sycl::atomic_ref<T,
                          sycl::memory_order::relaxed,
                          sycl::memory_scope::device,
                          sycl::access::address_space::global_space>
            atomic{acc[ValIdx]};

        acc[FetchMinStorageIdx] = atomic.fetch_min(MinValue, order); // returns 4, stores 3
        acc[FetchMaxStorageIdx] = atomic.fetch_max(MaxValueLast, order); // returns 3, stores 7
        atomic.fetch_min(InvalidMinValue, order); // remains 7; still an RMW
        atomic.fetch_max(InvalidMaxValue, order); // remains 7; still an RMW
      }
    );
   }).wait_and_throw();

  sycl::host_accessor result{buffer};

  BOOST_CHECK(result[ValIdx] == T{MaxValueLast}); // final after last successfull (fetch_max)
  BOOST_CHECK(result[FetchMinStorageIdx] == T{InitialValue}); //returns old value - 1st initial value
  BOOST_CHECK(result[FetchMaxStorageIdx] == T{MinValue}); //returns old value - 2nd (fetch_min)
}

BOOST_AUTO_TEST_CASE(floating_fetch_min_max_return_old_value) {
  sycl::queue q;
  // This regression targets the host atomic implementation. Device backend
  // lowering is tested independently.
  if(q.get_device().get_backend() != sycl::backend::omp) 
  {
    BOOST_TEST_MESSAGE("Skipping host atomic regression outside the OpenMP backend");
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
