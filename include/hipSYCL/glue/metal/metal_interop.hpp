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

#include "hipSYCL/sycl/backend.hpp"

#include <cstddef>

namespace hipsycl {
namespace sycl {

class context;

template <backend Backend>
struct AdaptiveCpp_native_allocation {};
template <backend Backend>
inline constexpr bool AdaptiveCpp_can_get_native_allocation = false;
template <backend Backend>
AdaptiveCpp_native_allocation<Backend> AdaptiveCpp_get_native_allocation(const void *, const context &) = delete;

} // namespace sycl
} // namespace hipsycl

#if defined(SYCL_EXT_ACPP_BACKEND_METAL) && defined(__APPLE__)
#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/metal/metal_allocator.hpp"
#include "hipSYCL/runtime/metal/metal_hardware_manager.hpp"

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

  static native_device_type get_native_device(const sycl::device &d) {
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
    auto *ctx = static_cast<rt::metal_hardware_context *>(
        hw->get_device(dev_id.get_id()));

    return ctx->get_mtl_device();
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
};

} // namespace glue

namespace sycl {

#ifdef ACPP_EXT_GET_NATIVE_ALLOCATION

template <> struct AdaptiveCpp_native_allocation<backend::metal> {
  MTL::Buffer *buffer;
  std::size_t offset;
};

template <>
inline constexpr bool AdaptiveCpp_can_get_native_allocation<backend::metal> = true;

// Returns the underlying Metal buffer and byte offset for a given SYCL USM pointer.
// If the pointer is not a recognized Metal USM pointer, returns {nullptr, 0}.
template <>
inline AdaptiveCpp_native_allocation<backend::metal>
AdaptiveCpp_get_native_allocation<backend::metal>(const void *ptr, const context &ctx) {
  rt::backend *b =
      ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::metal);
  if (!b)
    return {nullptr, 0};

  for (const device &dev : ctx.get_devices()) {
    rt::device_id dev_id = detail::extract_rt_device(dev);
    if (dev_id.get_backend() != rt::backend_id::metal)
      continue;

    rt::backend_allocator *base_alloc = b->get_allocator(dev_id);
    if (base_alloc) {
      auto *alloc = static_cast<rt::metal_allocator *>(base_alloc);
      auto [buf, offset, type] = alloc->get_usm_block(ptr);
      if (buf != nullptr) {
        return {buf, offset};
      }
    }
  }
  return {nullptr, 0};
}

#endif // ACPP_EXT_GET_NATIVE_ALLOCATION

} // namespace sycl
} // namespace hipsycl

#endif // defined(SYCL_EXT_ACPP_BACKEND_METAL) && defined(__APPLE__)
#endif // HIPSYCL_GLUE_METAL_BACKEND_INTEROP_HPP
