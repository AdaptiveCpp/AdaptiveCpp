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
#ifndef HIPSYCL_ACCESS_HPP
#define HIPSYCL_ACCESS_HPP

#include <ostream>

namespace hipsycl {
namespace sycl {

enum class target {
  device,
  host_task,
  global_buffer = device,
  // Deprecated targets
  constant_buffer = 2,
  local,
  image,
  host_buffer,
  host_image,
  image_array
};

enum class access_mode {
  read,
  write,
  read_write,
  // Deprecated access modes
  discard_write,
  discard_read_write,
  atomic
};


// TODO these should be moved to a common/serialization.hpp?
inline std::ostream &operator<<(std::ostream &out, const sycl::access_mode value)
{
  switch (value) {
  case sycl::access_mode::read:
    out << "R";
    break;
  case sycl::access_mode::write:
    out << "W";
    break;
  case sycl::access_mode::atomic:
    out << "atomic";
    break;
  case sycl::access_mode::read_write:
    out << "RW";
    break;
  case sycl::access_mode::discard_write:
    out << "Discard W";
    break;
  case sycl::access_mode::discard_read_write:
    out << "Discard RW";
    break;
  default:
    throw "Mode enum cannot be serialized";
    break;
  }
  return out;
}

inline std::ostream &operator<<(std::ostream &out,
                                const sycl::target value) {
  switch (value) {
  case sycl::target::image:
    out << "image";
    break;
  case sycl::target::constant_buffer:
    out << "constant_buffer";
    break;
  case sycl::target::device:
    out << "device";
    break;
  case sycl::target::host_buffer:
    out << "host_buffer";
    break;
  case sycl::target::host_image:
    out << "host_image";
    break;
  case sycl::target::image_array:
    out << "Image_array";
    break;
  case sycl::target::local:
    out << "local";
    break;
  case sycl::target::host_task:
    out << "host_task";
    break;
  default:
    throw "Target enum cannot be serialized";
    break;
  }
  return out;
}

// hipSYCL accessor variants:
// ranged placeholder: Stores mem_region, buff_range, range, offset
// ranged non-placeholder: Stores buff_range, range, offset
// non-ranged placeholder: Stores mem_region, buff_range
// non-ranged non-placeholder: Stores buff_range
// raw: nothing
enum class accessor_variant {
  false_t, // compatibility with SYCL 1.2.1 placeholder enum
  true_t,  // compatibility with SYCL 1.2.1 placeholder enum
  ranged_placeholder,
  ranged,
  unranged_placeholder,
  unranged,
  raw
};

namespace access {

// SYCL 1.2.1 compatibility
using sycl::target;
using mode = sycl::access_mode;

// Deprecated (reused in hipSYCL to store accessor variants)
using placeholder = sycl::accessor_variant;

enum class fence_space : char {
  local_space,
  global_space,
  global_and_local
};

enum class decorated : char {
  no,
  yes,
  legacy // [[deprecated]] TODO: This is deprecated but used as default value for multi_ptr template argument
};

inline std::ostream &operator<<(std::ostream &out,
                         const sycl::access::placeholder value)
{
  switch (value) {
  case sycl::access::placeholder::false_t:
    out << "false";
    break;
  case sycl::access::placeholder::true_t:
    out << "true";
    break;
  default:
    throw "Placeholder enum cannot be serialized";
    break;
  }
  return out;
}

inline std::ostream &operator<<(std::ostream &out,
                         const sycl::access::fence_space value) 
{
  switch (value) {
  case sycl::access::fence_space::global_and_local:
    out << "global and local";
    break;
  case sycl::access::fence_space::global_space:
    out << "global";
    break;
  case sycl::access::fence_space::local_space:
    out << "local";
    break;
  default:
    throw "fence_space enum cannot be serialized";
    break;
  }
  return out;
}

} // access
}
}


#endif
