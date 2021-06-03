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


#ifndef HIPSYCL_PLATFORM_HPP
#define HIPSYCL_PLATFORM_HPP

#include <vector>

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/device_id.hpp"

#include "types.hpp"
#include "device_selector.hpp"
#include "info/info.hpp"
#include "version.hpp"

namespace hipsycl {
namespace sycl {

class device_selector;

class platform {

public:
  platform() : _platform{detail::get_host_device().get_backend(), 0} {}
  
  platform(rt::backend_id backend)
      : _platform{backend, 0} {}

  template<class DeviceSelector>
  explicit platform(const DeviceSelector &deviceSelector) {
    auto dev = detail::select_devices(deviceSelector)[0];
    this->_platform = rt::platform_id{dev._device_id};
  }


  std::vector<device>
  get_devices(info::device_type type = info::device_type::all) const {
    std::vector<device> result;
    rt::backend *b = rt::application::backends().get(_platform.get_backend());
    
    int num_devices = b->get_hardware_manager()->get_num_devices();
    for (int dev = 0; dev < num_devices; ++dev) {
      bool is_cpu = b->get_hardware_manager()->get_device(dev)->is_cpu();
      bool is_gpu = b->get_hardware_manager()->get_device(dev)->is_gpu();

      bool include_device = false;
      if (type == info::device_type::all ||
          (type == info::device_type::accelerator && is_gpu) ||
          (type == info::device_type::gpu && is_gpu) ||
          (type == info::device_type::host && is_cpu) ||
          (type == info::device_type::cpu && is_cpu)) {
        include_device = true;
      }

      if (include_device)
        result.push_back(device{rt::device_id{b->get_backend_descriptor(), dev}});
    }
  
    return result;
  }


  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type get_info() const;


  /// \todo Think of a better solution
  bool has_extension(const string_class &extension) const {
    return false;
  }


  bool is_host() const {
    return rt::application::get_backend(_platform.get_backend())
               .get_backend_descriptor()
               .hw_platform == rt::hardware_platform::cpu;
  }

  /// Returns true if all devices in this platform have the
  /// specified aspect
  bool has(aspect asp) const {
    auto devs = get_devices();
    for(const device& d : devs) {
      if(!d.has(asp))
        return false;
    }
    return true;
  }

  static std::vector<platform> get_platforms() {
    std::vector<platform> result;
    rt::application::backends().for_each_backend([&](rt::backend *b) {
      result.push_back(platform{b->get_unique_backend_id()});
    });

    return result;
  }

  friend bool operator==(const platform &lhs, const platform &rhs) {
    return lhs._platform == rhs._platform;
  }

  friend bool operator!=(const platform &lhs, const platform &rhs) {
    return !(lhs == rhs);
  }

private:
  rt::platform_id _platform;
};


HIPSYCL_SPECIALIZE_GET_INFO(device, platform)
{ return this->get_platform(); }

HIPSYCL_SPECIALIZE_GET_INFO(platform, profile)
{ return "FULL_PROFILE"; }

HIPSYCL_SPECIALIZE_GET_INFO(platform, version)
{
  return detail::version_string();
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, name)
{
  rt::backend_id b = _platform.get_backend();
  return rt::application::get_backend(b).get_name();
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, vendor)
{
  return "The hipSYCL project";
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, extensions)
{
  return vector_class<string_class>{};
}

inline platform device::get_platform() const  {
  return platform{_device_id.get_backend()};
}

}// namespace sycl
}// namespace hipsycl

#endif
