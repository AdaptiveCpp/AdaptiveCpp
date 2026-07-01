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
#ifndef HIPSYCL_GLUE_METAL_BACKEND_INTEROP_HPP
#define HIPSYCL_GLUE_METAL_BACKEND_INTEROP_HPP

#ifdef SYCL_EXT_ACPP_BACKEND_METAL
#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/error.hpp"
#if defined(__APPLE__)
#include "hipSYCL/runtime/metal/metal_hardware_manager.hpp"
#endif

namespace MTL {
class Buffer;
class Device;
class CommandQueue;
} // namespace MTL

namespace hipsycl {
namespace glue {

template <> struct backend_interop<sycl::backend::metal> {
  using error_type = int;

  using native_mem_type = void *;
  using native_device_type = MTL::Device *;
  using native_queue_type = MTL::CommandQueue *;

  template <class Accessor_type>
  static native_mem_type get_native_mem(const Accessor_type &a) {
    return a.get_pointer();
  }

  struct native_allocation_type {
    MTL::Buffer *buffer; // Metal buffer underlying the allocation
    std::size_t offset; // Offset from the start of the buffer, in bytes
  };

  static native_device_type get_native_device(const sycl::device &d) {
#if defined(__APPLE__)
    rt::device_id dev_id = sycl::detail::extract_rt_device(d);

    if (dev_id.get_backend() != rt::backend_id::metal) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_device: device does not belong to the "
                         "Metal backend",
                         rt::error_type::invalid_parameter_error});
      return nullptr;
    }

    rt::backend *b =
        d.AdaptiveCpp_runtime()->backends().get(rt::backend_id::metal);
    if (!b) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_device: Metal backend not available",
                         rt::error_type::runtime_error});
      return nullptr;
    }

    auto *hw = static_cast<rt::metal_hardware_manager *>(b->get_hardware_manager());
    auto *ctx = static_cast<rt::metal_hardware_context *>(hw->get_device(dev_id.get_id()));

    return ctx->get_mtl_device();
#else
    rt::register_error(
        __acpp_here(),
        rt::error_info{"get_native_device: Metal backend not supported on this OS",
                       rt::error_type::runtime_error});
    return nullptr;
#endif
  }

  static native_queue_type get_native_queue(void *launcher_params) {
    if (!launcher_params) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_queue: invalid (null) launcher params",
                         rt::error_type::invalid_parameter_error});
      return nullptr;
    }

    return static_cast<native_queue_type>(
        static_cast<rt::inorder_queue *>(launcher_params)->get_native_type());
  }

  static native_allocation_type get_native_allocation(const void *ptr, const sycl::context &ctx) {
    rt::backend *b =
        ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::metal);
    if (!b)
      return {nullptr, 0};

    for (const sycl::device &dev : ctx.get_devices()) {
      rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
      if (dev_id.get_backend() != rt::backend_id::metal)
        continue;

      rt::backend_allocator *base_alloc = b->get_allocator(dev_id);
      if (base_alloc) {
        rt::pointer_info info;
        if (base_alloc->query_pointer(ptr, info).is_success() &&
            info.native_handle != nullptr) {
          return {static_cast<MTL::Buffer *>(info.native_handle),
                  info.native_offset};
        }
      }
    }
    return {nullptr, 0};
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
  static constexpr bool can_extract_native_device = true;
  static constexpr bool can_extract_native_context = false;
  // sycl::get_native(queue) is not implemented; use interop_handle::get_native_queue()
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
  static constexpr bool can_extract_native_allocation = true;
};

} // namespace glue
} // namespace hipsycl

#endif // SYCL_EXT_ACPP_BACKEND_METAL
#endif // HIPSYCL_GLUE_METAL_BACKEND_INTEROP_HPP
