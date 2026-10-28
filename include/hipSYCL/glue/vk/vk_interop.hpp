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
#pragma once

#ifdef SYCL_EXT_ACPP_BACKEND_VK
// Interop with the Vulkan backend requires the Vulkan headers to be available
// to user code. AdaptiveCpp does not add them to the include path, so only
// enable the interop API if they can be found.
#if defined(__has_include)
#if __has_include(<vulkan/vulkan_core.h>)
#define ACPP_GLUE_VK_INTEROP_AVAILABLE
// Feature test macro of the Vulkan backend interop API, which is only defined
// here as it depends on the Vulkan headers
#define ACPP_EXT_VK_INTEROP
#endif
#endif
#endif

#ifdef ACPP_GLUE_VK_INTEROP_AVAILABLE

#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/vk/vk_native_objects.hpp"

#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/event.hpp"
#include "hipSYCL/sycl/platform.hpp"

namespace hipsycl {
namespace glue {

namespace vk_interop_detail {

// Returns the native handles of the single Vulkan device contained in ctx.
// Interop requires an unambiguous VkDevice, so contexts containing more than
// one Vulkan device are rejected.
inline bool get_unique_device_handle(const sycl::context &ctx,
                                     rt::vk_device_native_handle &out,
                                     rt::device_id &out_dev) {
  rt::backend *b =
      ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    return false;
  }

  bool found = false;
  for (const sycl::device &dev : ctx.get_devices()) {
    rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
    if (dev_id.get_backend() != rt::backend_id::vk) {
      continue;
    }

    if (found) {
      // More than one Vulkan device in this context
      return false;
    }

    auto *provider = dynamic_cast<rt::vk_native_device_provider *>(
        b->get_hardware_manager()->get_device(dev_id.get_id()));
    if (!provider) {
      return false;
    }

    out = provider->get_native_device_handle();
    out_dev = dev_id;
    found = true;
  }

  return found;
}

inline bool get_device_handle(const sycl::device &d,
                              rt::vk_device_native_handle &out) {
  rt::device_id dev_id = sycl::detail::extract_rt_device(d);

  if (dev_id.get_backend() != rt::backend_id::vk) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{
            "vk interop: device does not belong to the Vulkan backend",
            rt::error_type::invalid_parameter_error});
    return false;
  }

  rt::backend *b = d.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
  if (!b) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{"vk interop: Vulkan backend not available",
                       rt::error_type::runtime_error});
    return false;
  }

  auto *provider = dynamic_cast<rt::vk_native_device_provider *>(
      b->get_hardware_manager()->get_device(dev_id.get_id()));
  if (!provider) {
    rt::register_error(
        __acpp_here(),
        rt::error_info{
            "vk interop: device does not expose native Vulkan objects",
            rt::error_type::runtime_error});
    return false;
  }

  out = provider->get_native_device_handle();
  return true;
}

// Turns an event that the backend has created from a native Vulkan event of
// the given device into a SYCL event.
inline sycl::event
make_imported_event(rt::runtime *runtime, rt::device_id dev,
                    std::shared_ptr<rt::dag_node_event> runtime_event) {
  auto op = std::make_unique<rt::kernel_operation>(
      "<native Vulkan semaphore>", rt::kernel_launcher({}, {}),
      rt::requirements_list{runtime});
  rt::dag_node_ptr node = std::make_shared<rt::dag_node>(
      rt::execution_hints{}, rt::node_list_t{}, std::move(op), runtime);
  node->assign_to_device(dev);
  node->assign_to_executor(nullptr);
  node->assign_to_execution_lane(nullptr);
  node->assign_execution_index(0);
  node->get_execution_hints().set_hint(rt::hints::instant_execution{});
  node->get_operation()->get_instrumentations().mark_set_complete();
  node->mark_submitted(std::move(runtime_event));

  return sycl::event{node};
}

} // namespace vk_interop_detail

template <> struct backend_interop<sycl::backend::vk> {

  using error_type = int;

  using native_mem_type = void *;
  using native_platform_type = VkInstance;
  using native_device_type = VkPhysicalDevice;
  using native_context_type = VkDevice;
  using native_event_type = rt::vk_event_native_handle;
  using native_allocation_type = rt::vk_allocation_native_handle;

  template <class Accessor_type>
  static native_mem_type get_native_mem(const Accessor_type &a) {
    return a.get_pointer();
  }

  static native_platform_type get_native_platform(const sycl::platform &p) {
    if (p.get_backend() != sycl::backend::vk) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_platform: platform does not belong to "
                         "the Vulkan backend",
                         rt::error_type::invalid_parameter_error});
      return VK_NULL_HANDLE;
    }

    std::vector<sycl::device> devices = p.get_devices();
    if (devices.empty()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{
              "get_native_platform: platform does not contain any devices",
              rt::error_type::runtime_error});
      return VK_NULL_HANDLE;
    }

    // All Vulkan devices of the backend originate from the same instance
    rt::vk_device_native_handle handle;
    if (!vk_interop_detail::get_device_handle(devices.front(), handle)) {
      return VK_NULL_HANDLE;
    }

    return handle.instance;
  }

  static native_device_type get_native_device(const sycl::device &d) {
    rt::vk_device_native_handle handle;
    if (!vk_interop_detail::get_device_handle(d, handle)) {
      return VK_NULL_HANDLE;
    }

    return handle.physical_device;
  }

  static native_context_type get_native_context(const sycl::context &ctx) {
    rt::vk_device_native_handle handle;
    rt::device_id dev_id;
    if (!vk_interop_detail::get_unique_device_handle(ctx, handle, dev_id)) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_context: context does not contain exactly "
                         "one Vulkan device",
                         rt::error_type::invalid_parameter_error});
      return VK_NULL_HANDLE;
    }

    return handle.device;
  }

  static native_event_type get_native_event(const sycl::event &e) {
    const native_event_type invalid_event{VK_NULL_HANDLE, 0};

    rt::dag_node_ptr node = sycl::detail::extract_rt_event(e);
    if (!node) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: invalid event",
                         rt::error_type::invalid_parameter_error});
      return invalid_event;
    }

    // The native synchronization point only exists once the operation has
    // been submitted to the backend.
    if (!node->is_submitted()) {
      rt::runtime_keep_alive_token{}.get()->dag().flush_and_gc();
    }

    if (!node->is_submitted()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has not been submitted",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    if (node->is_cancelled()) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has been cancelled",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    if (node->get_assigned_device().get_backend() != rt::backend_id::vk) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event does not belong to the "
                         "Vulkan backend",
                         rt::error_type::invalid_parameter_error});
      return invalid_event;
    }

    auto event = node->get_event();
    if (!event) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event has not been submitted",
                         rt::error_type::runtime_error});
      return invalid_event;
    }

    auto *vk_event = dynamic_cast<rt::vk_native_event_provider *>(event.get());
    if (!vk_event) {
      rt::register_error(
          __acpp_here(),
          rt::error_info{"get_native_event: event is not backed by a native "
                         "Vulkan timeline semaphore",
                         rt::error_type::feature_not_supported});
      return invalid_event;
    }

    return vk_event->get_native_event_handle();
  }

  static sycl::event make_sycl_event(const native_event_type &native_event,
                                     const sycl::context &ctx) {
    if (native_event.semaphore == VK_NULL_HANDLE) {
      throw sycl::exception{sycl::make_error_code(sycl::errc::invalid),
                            "make_event: invalid native Vulkan semaphore"};
    }

    rt::runtime *runtime = ctx.AdaptiveCpp_runtime();
    rt::backend *backend = runtime->backends().get(rt::backend_id::vk);
    if (!backend) {
      throw sycl::exception{sycl::make_error_code(sycl::errc::backend_mismatch),
                            "make_event: Vulkan backend not available"};
    }

    std::vector<rt::device_id> vk_devices;
    for (const sycl::device &dev : ctx.get_devices()) {
      rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
      if (dev_id.get_backend() == rt::backend_id::vk) {
        vk_devices.push_back(dev_id);
      }
    }

    if (vk_devices.empty()) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::backend_mismatch),
          "make_event: the supplied context does not contain a Vulkan device"};
    }

    if (native_event.device == VK_NULL_HANDLE && vk_devices.size() > 1) {
      throw sycl::exception{
          sycl::make_error_code(sycl::errc::invalid),
          "make_event: the VkDevice of the native Vulkan event must be set, as "
          "the supplied context contains more than one Vulkan device"};
    }

    // The backend only accepts the native event for the device it belongs to
    for (rt::device_id dev : vk_devices) {
      std::shared_ptr<rt::dag_node_event> runtime_event =
          backend->create_event_from_native_handle(&native_event, dev);
      if (runtime_event) {
        return vk_interop_detail::make_imported_event(runtime, dev,
                                                      std::move(runtime_event));
      }
    }

    throw sycl::exception{
        sycl::make_error_code(sycl::errc::invalid),
        "make_event: the native Vulkan event does not belong to a device of "
        "the supplied context"};
  }

  static native_allocation_type
  get_native_allocation(const void *ptr, const sycl::context &ctx) {
    const native_allocation_type invalid_allocation{VK_NULL_HANDLE, 0, 0};

    rt::backend *b =
        ctx.AdaptiveCpp_runtime()->backends().get(rt::backend_id::vk);
    if (!b) {
      return invalid_allocation;
    }

    for (const sycl::device &dev : ctx.get_devices()) {
      rt::device_id dev_id = sycl::detail::extract_rt_device(dev);
      if (dev_id.get_backend() != rt::backend_id::vk) {
        continue;
      }

      auto *provider = dynamic_cast<rt::vk_native_allocation_provider *>(
          b->get_allocator(dev_id));
      if (!provider) {
        continue;
      }

      native_allocation_type result;
      if (provider->get_native_allocation(ptr, result)) {
        return result;
      }
    }

    return invalid_allocation;
  }

  static constexpr bool can_make_platform = false;
  static constexpr bool can_make_device = false;
  static constexpr bool can_make_context = false;
  static constexpr bool can_make_queue = false;
  static constexpr bool can_make_event = true;
  static constexpr bool can_make_buffer = false;
  static constexpr bool can_make_sampled_image = false;
  static constexpr bool can_make_image_sampler = false;
  static constexpr bool can_make_stream = false;
  static constexpr bool can_make_kernel = false;
  static constexpr bool can_make_module = false;

  static constexpr bool can_extract_native_platform = true;
  static constexpr bool can_extract_native_device = true;
  static constexpr bool can_extract_native_context = true;
  // sycl::get_native(queue) is not implemented, since AdaptiveCpp may use a
  // different backend queue for each operation.
  static constexpr bool can_extract_native_queue = false;
  static constexpr bool can_extract_native_event = true;
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

#endif // ACPP_GLUE_VK_INTEROP_AVAILABLE
