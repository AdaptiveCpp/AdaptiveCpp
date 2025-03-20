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
#include <iostream>
#include <pcuda.hpp>
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(pcuda_device_query)

class reset_device {
public:
  reset_device() {
    if(pcudaGetBackend(&_b) != pcudaSuccess)
      _initialization_successful = false;
    if(pcudaGetPlatform(&_p) != pcudaSuccess)
      _initialization_successful = false;
    if(pcudaGetDevice(&_d) != pcudaSuccess)
      _initialization_successful = false;
  }

  ~reset_device() {
    pcudaSetDeviceExt(_b, _p, _d);
  }
private:
  int _b;
  int _p;
  int _d;

  bool _initialization_successful = true;
};

BOOST_AUTO_TEST_CASE(enumeration) {
  // This test may change the active device; this guard
  // ensures that the device is reset to the original one
  // so as to not perturb subsequent tests.
  reset_device r;

  int backend_count = 0;
  BOOST_CHECK(pcudaGetBackendCount(&backend_count) == pcudaSuccess);
  BOOST_CHECK(backend_count > 0);

  int total_device_count = 0;
  
  for(int backend = 0; backend < backend_count; ++backend) {
    std::cout << "Backend " << backend << std::endl;

    auto err = pcudaSetBackend(backend);
    BOOST_CHECK(err == pcudaSuccess || err == pcudaErrorNoDevice);
    
    int platform_count = 0;
    err = pcudaGetPlatformCount(&platform_count);
    BOOST_CHECK(err == pcudaSuccess || err == pcudaErrorNoDevice);
    for(int platform = 0; platform < platform_count; ++platform) {
      std::cout << " Platform " << platform << std::endl;
      err = pcudaSetPlatform(platform);

      BOOST_CHECK(err == pcudaSuccess || err == pcudaErrorNoDevice);

      int device_count = 0;
      err = pcudaGetDeviceCount(&device_count);
      BOOST_CHECK(err == pcudaSuccess || err == pcudaErrorNoDevice);

      for(int device = 0; device < device_count; ++device) {
        std::cout << "  Device " << device << std::endl;
        ++total_device_count;
        err = pcudaSetDevice(device);
        BOOST_CHECK(err == pcudaSuccess);

        int retrieved_device = 0;
        BOOST_CHECK(pcudaGetDevice(&retrieved_device) == pcudaSuccess);
        BOOST_CHECK(retrieved_device == device);
      }
    }
  }
  BOOST_CHECK(total_device_count > 0);
}


BOOST_AUTO_TEST_SUITE_END()
