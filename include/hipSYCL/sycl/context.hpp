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

#include "types.hpp"
#include "platform.hpp"
#include "exception.hpp"
#include "device.hpp"
#include "info/info.hpp"

#include <cassert>

namespace hipsycl {
namespace sycl {


class context
{
public:
  explicit context(async_handler asyncHandler = {})
  {}

  context(const device &dev, async_handler asyncHandler = {})
    : _platform{dev.get_platform()}, _devices{dev}
  {}

  context(const platform &plt, async_handler asyncHandler = {})
    : _platform{plt}, _devices(plt.get_devices())
  {}

  context(const vector_class<device> &deviceList,
          async_handler asyncHandler = {})
    : _devices{deviceList}
  {
    if(deviceList.empty())
      throw platform_error{"context: Could not infer platform from empty device list"};

    _platform = deviceList.front().get_platform();

    for(const auto dev : deviceList) {
      (void)(&dev);
      assert(dev.get_platform() == _platform);
    }
  }

  /* CL Interop is not supported
  context(cl_context clContext, async_handler asyncHandler = {});
  */


  /* -- common interface members -- */


  /* CL interop is not supported
  cl_context get() const;
*/

  bool is_host() const {
#ifdef HIPSYCL_PLATFORM_CPU
    return true;
#else
    return false;
#endif
  }

  platform get_platform() const {
    return _platform;
  }

  vector_class<device> get_devices() const {
    return _devices;
  }

  template <info::context param>
  typename info::param_traits<info::context, param>::return_type get_info() const {
    throw unimplemented{"context::get_info() is unimplemented"};
  }

private:
  platform _platform;
  vector_class<device> _devices;
};


HIPSYCL_SPECIALIZE_GET_INFO(context, reference_count)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(context, platform)
{ return get_platform(); }

HIPSYCL_SPECIALIZE_GET_INFO(context, devices)
{ return get_devices(); }

} // namespace sycl
} // namespace hipsycl



#endif
