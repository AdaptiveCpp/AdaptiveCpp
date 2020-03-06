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
  virtual ~device_selector() {};
  device select_device() const;

  virtual int operator()(const device& dev) const = 0;

};

namespace detail {
class select_all_selector : public device_selector
{
public:
  virtual ~select_all_selector(){}
  virtual int operator()(const device &dev) const { return 1; }
};

} // namespace detail

class error_selector : public device_selector
{
public:
  virtual ~error_selector(){}
  virtual int operator()(const device& dev) const
  {
    throw unimplemented{"hipSYCL presently only supports GPU platforms when using the CUDA and ROCm "
                        "backends, and CPU platforms when compiling against hipCPU"};
  }
};

#ifdef __HIPCPU__
using gpu_selector = error_selector;
#else
using gpu_selector = detail::select_all_selector;
#endif

#ifdef __HIPCPU__
using cpu_selector = detail::select_all_selector;
#else
using cpu_selector = error_selector;
#endif

using host_selector = cpu_selector;

#ifdef __HIPCPU__
using default_selector = host_selector;
#else
using default_selector = gpu_selector;
#endif

}
}

#endif
