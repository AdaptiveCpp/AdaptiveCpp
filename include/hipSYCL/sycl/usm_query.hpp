/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "context.hpp"
#include "device.hpp"
#include "exception.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include <exception>

#ifndef HIPSYCL_USM_QUERY_HPP
#define HIPSYCL_USM_QUERY_HPP

namespace hipsycl {
namespace sycl {


namespace detail {


inline rt::backend_id select_usm_backend(const context &ctx) {
  const rt::unique_device_list &devs = detail::extract_context_devices(ctx);

  if (devs.get_num_backends() == 0)
    throw memory_allocation_error{
        "USM: No backends to carry out USM memory management are present in "
        "the context!"};


  std::size_t num_host_backends =
      devs.get_num_backends(rt::hardware_platform::cpu);

  assert(devs.get_num_backends() >= num_host_backends);
  std::size_t num_device_backends = devs.get_num_backends() - num_host_backends;

  rt::backend_id selected_backend;

  // Logic is simple: Prefer device backends if available over host backends.
  if (num_device_backends > 0) {
    if (num_device_backends > 1) {
      HIPSYCL_DEBUG_WARNING
          << "USM backend selection: Context contains multiple device "
             "backends. "
             "Using the first device backend as GUESS for USM memory "
             "management. This might go TERRIBLY WRONG if you are not using "
             "this backend for your kernels! "
             "You are encouraged to better specify which backends you want to "
             "target."
          << std::endl;
    }
    auto backend_it = devs.find_first_backend([](rt::backend_id b) {
      return rt::application::get_backend(b).get_hardware_platform() !=
             rt::hardware_platform::cpu;
    });
    assert(backend_it != devs.backends_end());
    selected_backend = *backend_it;

  } else if (num_host_backends > 0) {
    if (num_host_backends > 1) {
      HIPSYCL_DEBUG_WARNING
          << "USM backend selection: Context did not contain any device "
             "backends, but multiple host backends. Using first host backend "
             "(which should be fine as long as you run only on the host), but "
             "you are encouraged to better specify which backend you wish to "
             "carry out USM memory management"
          << std::endl;
    }
    auto backend_it = devs.find_first_backend([](rt::backend_id b) {
      return rt::application::get_backend(b).get_hardware_platform() ==
             rt::hardware_platform::cpu;
    });
    assert(backend_it != devs.backends_end());
    selected_backend = *backend_it;
  } else {
    // Prevent warnings about uninitialized use of selected_backend
    selected_backend = detail::get_host_device().get_backend();
    throw memory_allocation_error{
        "USM: Could not select backend to use for USM memory operation"};
  }
  
  return selected_backend;
}

inline rt::backend_allocator *select_usm_allocator(const context &ctx) {
  rt::backend_id selected_backend = select_usm_backend(ctx);

  rt::backend &backend_object = rt::application::get_backend(selected_backend);

  if (backend_object.get_hardware_manager()->get_num_devices() == 0)
    throw memory_allocation_error{"USM: Context has no devices on which "
                                  "requested operation could be carried out"};

  return backend_object.get_allocator(
      rt::device_id{backend_object.get_backend_descriptor(), 0});
}

inline rt::backend_allocator *select_usm_allocator(const context &ctx,
                                                   const device &dev) {
  rt::backend_id selected_backend = select_usm_backend(ctx);

  rt::backend &backend_object = rt::application::get_backend(selected_backend);
  rt::device_id d = detail::extract_rt_device(dev);
  
  if(d.get_backend() == selected_backend)
    return backend_object.get_allocator(detail::extract_rt_device(dev));
  else
    return backend_object.get_allocator(
        rt::device_id{d.get_full_backend_descriptor(), 0});
}

inline rt::backend_allocator *select_device_allocator(const device &dev) {
  rt::device_id d = detail::extract_rt_device(dev);

  rt::backend& backend_object = rt::application::get_backend(d.get_backend());
  return backend_object.get_allocator(d);
}

}

namespace usm {
enum class alloc { host, device, shared, unknown };
}


inline usm::alloc get_pointer_type(const void *ptr, const context &ctx) {
  rt::pointer_info info;
  rt::result res = detail::select_usm_allocator(ctx)->query_pointer(ptr, info);

  if (!res.is_success())
    return usm::alloc::unknown;

  if (info.is_from_host_backend || info.is_optimized_host)
    return usm::alloc::host;
  else if (info.is_usm)
    return usm::alloc::shared;
  else
    return usm::alloc::device;
}

inline sycl::device get_pointer_device(const void *ptr, const context &ctx) {
  rt::pointer_info info;
  rt::result res = detail::select_usm_allocator(ctx)->query_pointer(ptr, info);

  if (!res.is_success())
    std::rethrow_exception(glue::throw_result(res));

  if (info.is_from_host_backend){
    // TODO Spec says to return *the* host device, but it might be better
    // to return a device from the actual host backend used
    // (we might want to have multiple host devices/backends in the future)
    return detail::get_host_device();
  }
  else if (info.is_optimized_host) {
    // Return first (non-host?) device from context
    const rt::unique_device_list &devs = detail::extract_context_devices(ctx);

    auto device_iterator =
        devs.find_first_device([](rt::device_id d) { return !d.is_host(); });

    assert(device_iterator != devs.devices_end());
    return device{*device_iterator};
  }
  else
    return device{info.dev};
}

}
} // namespace hipsycl

#endif
