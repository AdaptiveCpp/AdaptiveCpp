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
 
#ifndef HIPSYCL_TESTS_RESET_HPP
#define HIPSYCL_TESTS_RESET_HPP

struct reset_device_fixture {
  ~reset_device_fixture() {
    // Runtime should automatically reset
  }
};

#endif
