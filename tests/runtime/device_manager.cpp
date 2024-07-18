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

#include "runtime_test_suite.hpp"

#include <future>

#include <SYCL/sycl.hpp>
#include <hipSYCL/runtime/device_id.hpp>

#ifdef HIPSYCL_PLATFORM_CUDA
#include <hipSYCL/runtime/cuda/cuda_backend.hpp>
#include <hipSYCL/runtime/cuda/cuda_device_manager.hpp>
#endif

#ifdef HIPSYCL_PLATFORM_HIP
#include <hipSYCL/runtime/hip/hip_backend.hpp>
#include <hipSYCL/runtime/hip/hip_device_manager.hpp>
#endif

using namespace hipsycl;

template <rt::backend_id Backend> struct backend_enabled {
  static constexpr bool value = false;
};

template <rt::backend_id Backend> struct backend_class { using type = void; };

template <rt::backend_id Backend> struct backend_device_manager {
  using type = void;
};

#ifdef HIPSYCL_PLATFORM_CUDA
template <> struct backend_enabled<rt::backend_id::cuda> {
  static constexpr bool value = true;
};

template <> struct backend_class<rt::backend_id::cuda> {
  using type = rt::cuda_backend;
};

template <> struct backend_device_manager<rt::backend_id::cuda> {
  using type = rt::cuda_device_manager;
};
#endif

#ifdef HIPSYCL_PLATFORM_HIP
template <> struct backend_enabled<rt::backend_id::hip> {
  static constexpr bool value = true;
};

template <> struct backend_class<rt::backend_id::hip> {
  using type = rt::hip_backend;
};

template <> struct backend_device_manager<rt::backend_id::hip> {
  using type = rt::hip_device_manager;
};
#endif

BOOST_FIXTURE_TEST_SUITE(device_manager, reset_device_fixture)

namespace btt = boost::test_tools;

template <rt::backend_id Backend>
struct if_backend_and_devices_available {
  btt::assertion_result operator()(boost::unit_test::test_unit_id) {
    btt::assertion_result ans(false);
    if constexpr (!backend_enabled<Backend>::value) {
      ans.message() << "backend is not enabled.";
      return ans;
    } else {
      typename backend_class<Backend>::type backend;
      if (backend.get_hardware_manager()->get_num_devices() < 2) {
        ans.message() << "at least two devices are required.";
        return ans;
      }
    }
    return true;
  }
};

template <rt::backend_id Backend>
void run_device_manager_multithreaded_test() {
  if constexpr (backend_enabled<Backend>::value) {
    using mngr = typename backend_device_manager<Backend>::type;

    mngr::get().activate_device(1);
    BOOST_CHECK(mngr::get().get_active_device() == 1);

    std::async(std::launch::async, [] {
      BOOST_CHECK(mngr::get().get_active_device() == 0);
      mngr::get().activate_device(0);
    }).wait();

    BOOST_CHECK(mngr::get().get_active_device() == 1);
  }
}

BOOST_AUTO_TEST_CASE(
    cuda_device_manager_multithreaded,
    *boost::unit_test::precondition(
        if_backend_and_devices_available<rt::backend_id::cuda>{})) {
  run_device_manager_multithreaded_test<rt::backend_id::cuda>();
}

BOOST_AUTO_TEST_CASE(
    hip_device_manager_multithreaded,
    *boost::unit_test::precondition(
        if_backend_and_devices_available<rt::backend_id::hip>{})) {
  run_device_manager_multithreaded_test<rt::backend_id::hip>();
}

BOOST_AUTO_TEST_SUITE_END()
