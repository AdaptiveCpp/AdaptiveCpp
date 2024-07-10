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
#ifndef HIPSYCL_INFO_QUEUE_HPP
#define HIPSYCL_INFO_QUEUE_HPP

#include "info.hpp"
#include "../types.hpp"

namespace hipsycl {
namespace sycl {

class context;
class device;

namespace info {

namespace queue
{
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(context, sycl::context);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(device, sycl::device);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(reference_count, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(AdaptiveCpp_node_group, std::size_t);
};

}
}
}

#endif
