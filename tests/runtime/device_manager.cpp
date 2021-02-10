/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020-2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
/*
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
*/