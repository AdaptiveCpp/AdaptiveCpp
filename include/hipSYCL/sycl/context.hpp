/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_CONTEXT_HPP
#define HIPSYCL_CONTEXT_HPP

#include <cassert>
#include <memory>
#include <vector>

#include "types.hpp"
#include "platform.hpp"
#include "exception.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "info/info.hpp"

#include "hipSYCL/runtime/device_list.hpp"
#include "hipSYCL/glue/error.hpp"

namespace hipsycl {
namespace sycl {

class context;

namespace detail {
const rt::unique_device_list& extract_context_devices(const context&);
}

class context
{
public:
  friend class queue;

  friend const rt::unique_device_list &
  detail::extract_context_devices(const context &);

  explicit context(async_handler handler = [](exception_list e) {
    glue::default_async_handler(e);
  }) : context{detail::select_devices(default_selector_v), handler} {}

  explicit context(
      const device &dev, async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    this->init(handler, dev);
  }

  explicit context(
      const platform &plt, async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    this->init(handler);
    std::vector<device> devices = plt.get_devices();
    for (const auto &dev : devices) {
      _impl->devices.add(dev._device_id);
    }
    // Always need to add the host device
    _impl->devices.add(detail::get_host_device());
  }

  explicit context(
      const std::vector<device> &deviceList,
      async_handler handler = [](exception_list e) {
        glue::default_async_handler(e);
      }) {
    
    if(deviceList.empty())
      throw platform_error{"context: Cannot construct context for empty device list"};

    this->init(handler);
    for(const device& dev : deviceList) {
      _impl->devices.add(dev._device_id);
    }
    // Always need to add the host device
    _impl->devices.add(detail::get_host_device());
  }

  bool is_host() const {
    bool has_non_host_devices = false;
    _impl->devices.for_each_device([&](rt::device_id d) {
      if (!d.is_host())
        has_non_host_devices = true;
    });
    return !has_non_host_devices;
  }

  platform get_platform() const {
    bool found_device_backend = false;
    rt::backend_id last_backend;

    this->_impl->devices.for_each_backend([&](rt::backend_id b) {
      if (b != detail::get_host_device().get_backend()) {
        if (found_device_backend) {
          // We already have a device backend
          HIPSYCL_DEBUG_WARNING
              << "context: get_platform() was called but this context spans "
                 "multiple backends/platforms. Only returning last platform"
              << std::endl;
        }
        
        last_backend = b;
        found_device_backend = true;
      }
    });

    if (!found_device_backend) {
      last_backend = detail::get_host_device().get_backend(); 
    }

    return platform{last_backend};
  }

  vector_class<device> get_devices() const {
    std::vector<device> devs;
    _impl->devices.for_each_device([&](rt::device_id d) {
      devs.push_back(d);
    });
    return devs;
  }

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type get_info() const {
    throw unimplemented{"context::get_info() is unimplemented"};
  }

private:
  void init(async_handler handler) {
    _impl = std::make_shared<context_impl>();
    _impl->handler = handler;
  }

  void init(async_handler handler, const device &d) {
    init(handler);
    _impl->devices.add(d._device_id);
    if(!d.is_host()) {
      // Always need to add the host device
      _impl->devices.add(detail::get_host_device());
    }
  }
  
  struct context_impl {
    rt::unique_device_list devices;
    async_handler handler;
  };

  std::shared_ptr<context_impl> _impl;
};


HIPSYCL_SPECIALIZE_GET_INFO(context, reference_count)
{ return _impl.use_count(); }

HIPSYCL_SPECIALIZE_GET_INFO(context, platform)
{ return get_platform(); }

HIPSYCL_SPECIALIZE_GET_INFO(context, devices)
{ return get_devices(); }

inline context exception::get_context() const {
  // ToDo In hipSYCL, most operations are not associated
  // with a context at all, so just return empty one?
  return context{};
}

namespace detail {

inline const rt::unique_device_list &extract_context_devices(const context &ctx) {
  return ctx._impl->devices;
}

}

} // namespace sycl
} // namespace hipsycl



#endif
