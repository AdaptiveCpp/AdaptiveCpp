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
#include "hipSYCL/sycl/backend.hpp"
#include "hipSYCL/sycl/device.hpp"


#include "hipSYCL/runtime/error.hpp"

#ifndef HIPSYCL_GLUE_ZE_BACKEND_INTEROP_HPP
#define HIPSYCL_GLUE_ZE_BACKEND_INTEROP_HPP

struct _ze_device_handle_t;
struct _ze_command_list_handle_t;

namespace hipsycl {
namespace glue {

template <> struct backend_interop<sycl::backend::level_zero> {
  
  using error_type = int;

  using native_mem_type = void *;
  using native_device_type = _ze_device_handle_t*;
  using native_queue_type = _ze_command_list_handle_t*;

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

}
}

#endif
