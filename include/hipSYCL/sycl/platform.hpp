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

#include "types.hpp"
#include "device_selector.hpp"
#include "info/info.hpp"
#include "version.hpp"

namespace hipsycl {
namespace sycl {

class device_selector;

class platform {

public:

  platform() {}

  /* OpenCL interop is not supported
  explicit platform(cl_platform_id platformID);
  */

  explicit platform(const device_selector &deviceSelector) {}


  /* -- common interface members -- */

  /* OpenCL interop is not supported
  cl_platform_id get() const;
  */


  vector_class<device> get_devices(
      info::device_type type = info::device_type::all) const
  {
    return device::get_devices();
  }


  template <info::platform param>
  typename info::param_traits<info::platform, param>::return_type get_info() const;


  /// \todo Think of a better solution
  bool has_extension(const string_class &extension) const {
    return false;
  }


  bool is_host() const {
    return false;
  }


  static vector_class<platform> get_platforms() {
    return vector_class<platform>{platform()};
  }

  friend bool operator==(const platform& lhs, const platform& rhs)
  { return true; }

  friend bool operator!=(const platform& lhs, const platform& rhs)
  { return !(lhs == rhs); }
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
#ifdef HIPSYCL_PLATFORM_CUDA
  return "hipSYCL [SYCL over CUDA/HIP] on NVIDIA CUDA";
#elif defined HIPSYCL_PLATFORM_HCC
  return "hipSYCL [SYCL over CUDA/HIP] on AMD ROCm";
#else
  return "hipSYCL [SYCL over CUDA/HIP] on hipCPU host device";
#endif
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, vendor)
{
  return "The hipSYCL project";
}

HIPSYCL_SPECIALIZE_GET_INFO(platform, extensions)
{
  return vector_class<string_class>{};
}

}// namespace sycl
}// namespace hipsycl

#endif
