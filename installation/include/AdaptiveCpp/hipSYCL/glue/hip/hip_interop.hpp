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

#ifdef SYCL_EXT_HIPSYCL_BACKEND_HIP
#include "hipSYCL/runtime/hip/hip_queue.hpp"
#include "hipSYCL/runtime/error.hpp"

#ifndef HIPSYCL_GLUE_HIP_BACKEND_INTEROP_HPP
#define HIPSYCL_GLUE_HIP_BACKEND_INTEROP_HPP

// Forward declare so that we do not need to rely on including HIP headers.
// This wouldn't work in certain explicit multipass configurations, e.g. in CUDA
// passes where the HIP headers cannot be included.
class ihipStream_t;

namespace hipsycl {
namespace glue {


template <> struct backend_interop<sycl::backend::hip> {
  // Use int instead of hipError_t to avoid having to include HIP
  // headers
  using error_type = int;

  using native_mem_type = void *;
  using native_device_type = int;
  using native_queue_type = ihipStream_t*;

  template <class Accessor_type>
  static native_mem_type get_native_mem(const Accessor_type &a) {
    return a.get_pointer();
  }

  static native_device_type get_native_device(const sycl::device &d) {
    return sycl::detail::extract_rt_device(d).get_id();
  }

  static native_queue_type
  get_native_queue(void *launcher_params) {

    if (!launcher_params) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"Invalid argument to get_native_queue()"});
      
      return native_queue_type{};
    }

    rt::inorder_queue* q = static_cast<rt::inorder_queue*>(launcher_params);
    return static_cast<native_queue_type>(q->get_native_type());
  }

  static native_queue_type
  get_native_queue(rt::device_id dev, rt::backend_executor *executor) {
    rt::multi_queue_executor *mqe =
        dynamic_cast<rt::multi_queue_executor *>(executor);

    if (!mqe) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"Invalid argument to get_native_queue()"});
      return native_queue_type{};
    }

    rt::inorder_queue *q = nullptr;
    mqe->for_each_queue(
        dev, [&](rt::inorder_queue *current_queue) { q = current_queue; });
    assert(q);

    return static_cast<native_queue_type>(q->get_native_type());
  }


  static sycl::device make_sycl_device(int device_id) {
    return sycl::device{
        rt::device_id{rt::backend_descriptor{rt::hardware_platform::rocm,
                                             rt::api_platform::hip},
                      device_id}};
  }

  static constexpr bool can_make_platform = false;
  static constexpr bool can_make_device = true;
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
  static constexpr bool can_extract_native_device = true;
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
#endif
