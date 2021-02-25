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


#ifndef HIPSYCL_DEVICE_SELECTOR_HPP
#define HIPSYCL_DEVICE_SELECTOR_HPP

#include "exception.hpp"
#include "device.hpp"

#include <limits>

namespace hipsycl {
namespace sycl {


class device_selector
{
public:
  virtual ~device_selector(){};
  
  device select_device() const {
    auto devices = device::get_devices();
    if (devices.size() == 0)
      throw platform_error{"No available devices!"};

    int best_score = std::numeric_limits<int>::min();
    device candidate;
    for (const device &d : devices) {
      int current_score = (*this)(d);
      if (current_score > best_score) {
        best_score = current_score;
        candidate = d;
      }
    }
    return candidate;
  }

  virtual int operator()(const device& dev) const = 0;

};


class error_selector : public device_selector
{
public:
  virtual ~error_selector(){}
  virtual int operator()(const device& dev) const
  {
    throw unimplemented{"error_selector device selection invoked"};
  }
};

class gpu_selector : public device_selector
{
public:
  virtual ~gpu_selector() {}
  virtual int operator()(const device &dev) const {
    if (dev.is_gpu()) {
      // Would be good to prefer a device for which
      // we have actually compiled kernel code, because,
      // I don't know, a user might try to run kernels..
      if (dev.hipSYCL_has_compiled_kernels())
        return 2;
      else
        return 1;
      
    }
    return 0;
  }
};

class cpu_selector : public device_selector
{
public:
  virtual ~cpu_selector() {}
  virtual int operator()(const device &dev) const {
    return dev.is_cpu();
  }
};

using host_selector = cpu_selector;

#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__) ||                                 \
    defined(__HIPSYCL_ENABLE_HIP_TARGET__) ||                                  \
    defined(__HIPSYCL_ENABLE_SPIRV_TARGET__)
using default_selector = gpu_selector;
#else
using default_selector = host_selector;
#endif

inline device::device(const device_selector &deviceSelector) {
  this->_device_id = deviceSelector.select_device()._device_id;
}


}
}

#endif
