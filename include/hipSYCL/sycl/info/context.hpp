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
#ifndef HIPSYCL_INFO_CONTEXT_HPP
#define HIPSYCL_INFO_CONTEXT_HPP

#include "info.hpp"
#include "../types.hpp"
#include "../libkernel/memory.hpp"

namespace hipsycl {
namespace sycl {

class platform;
class device;

namespace info {

namespace context {
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(reference_count, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(platform, sycl::platform);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(devices, std::vector<sycl::device>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(atomic_memory_order_capabilities,
                                 std::vector<memory_order>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(atomic_memory_scope_capabilities,
                                 std::vector<memory_scope>);
};

}
}
}

#endif
