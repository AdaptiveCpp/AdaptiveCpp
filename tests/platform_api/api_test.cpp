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
#include <sycl/sycl.hpp>

int main()
{
  sycl::platform platform;
  auto devs = platform.get_devices();

  for(const auto& d : devs)
    std::cout << "Found device "
              << d.get_info<sycl::info::device::name>()
              << std::endl;

  sycl::queue q;
  std::cout << "Created queue on GPU: " << ((q.is_host() == false) ? "true" : "false")
                                        << std::endl;
}
