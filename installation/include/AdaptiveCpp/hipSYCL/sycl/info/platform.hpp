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
#ifndef HIPSYCL_INFO_PLATFORM_HPP
#define HIPSYCL_INFO_PLATFORM_HPP

#include "info.hpp"
#include "../types.hpp"

namespace hipsycl {
namespace sycl {
namespace info {

namespace platform
{
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(profile, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(version, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(name, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(vendor, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(extensions, vector_class<string_class>);
};

}
}
}

#endif
