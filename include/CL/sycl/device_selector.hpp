/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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


#ifndef SYCU_DEVICE_SELECTOR_HPP
#define SYCU_DEVICE_SELECTOR_HPP

#include "exception.hpp"
#include "device.hpp"

#include <limits>

namespace cl {
namespace sycl {


class device_selector
{
public:
  virtual ~device_selector();
  device select_device() const;

  virtual int operator()(const device& dev) const = 0;

};


class gpu_selector : public device_selector
{
public:
  virtual ~gpu_selector(){}
  virtual int operator()(const device& dev) const
  {
    return 1;
  }
};

class error_selector : public device_selector
{
public:
  virtual ~error_selector(){}
  virtual int operator()(const device& dev) const
  {
    throw unimplemented{"SYCU presently only supports GPU platforms and device selectors."};
  }
};

using default_selector = gpu_selector;
using cpu_selector = error_selector;
using host_selector = error_selector;

}  // namespace sycl
}  // namespace cl

#endif
