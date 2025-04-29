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
#ifndef HIPSYCL_GLUE_OMPHOST_BACKEND_INTEROP_HPP
#define HIPSYCL_GLUE_OMPHOST_BACKEND_INTEROP_HPP

#include "hipSYCL/sycl/backend.hpp"

namespace hipsycl {
namespace glue {

#ifdef SYCL_EXT_HIPSYCL_BACKEND_OMPHOST
template <> struct backend_interop<sycl::backend::omp> {
  // Well, there's not a really a native error code type
  using error_type = int;

  using native_mem_type = void *;

  template <class Accessor_type>
  static native_mem_type get_native_mem(const Accessor_type &a) {
    return a.get_pointer();
  }

  static constexpr bool can_make_platform = false;
  static constexpr bool can_make_device = false;
  static constexpr bool can_make_context = false;
  static constexpr bool can_make_queue = false;
  static constexpr bool can_make_event = false;
  static constexpr bool can_make_buffer = false;
  static constexpr bool can_make_sampled_image = false;
  static constexpr bool can_make_image_sampler = false;
  static constexpr bool can_make_stream = false;
  static constexpr bool can_make_kernel = false;
  static constexpr bool can_make_module = false;

  static constexpr bool can_extract_native_platform = false;
  static constexpr bool can_extract_native_device = false;
  static constexpr bool can_extract_native_context = false;
  static constexpr bool can_extract_native_queue = false;
  static constexpr bool can_extract_native_event = false;
  static constexpr bool can_extract_native_buffer = false;
  static constexpr bool can_extract_native_sampled_image = false;
  static constexpr bool can_extract_native_image_sampler = false;
  static constexpr bool can_extract_native_stream = false;
  static constexpr bool can_extract_native_kernel = false;
  static constexpr bool can_extract_native_module = false;
  static constexpr bool can_extract_native_device_event = false;
  static constexpr bool can_extract_native_mem = true;
};
#endif

}
}

#endif
