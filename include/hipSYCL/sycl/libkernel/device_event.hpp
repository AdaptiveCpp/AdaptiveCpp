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
#ifndef HIPSYCL_DEVICE_EVENT_HPP
#define HIPSYCL_DEVICE_EVENT_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

namespace hipsycl {
namespace sycl {

class device_event
{
public:
  ACPP_KERNEL_TARGET
  device_event(){}

  ACPP_KERNEL_TARGET
  void wait(){}
};

}
}


#endif
