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
#include <functional>
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

inline int select_gpu(const device& dev) {
  if (dev.is_gpu()) {
    // Would be good to prefer a device for which
    // we have actually compiled kernel code, because,
    // I don't know, a user might try to run kernels..
    if (dev.hipSYCL_has_compiled_kernels())
      return 2;
    else
      return 1;
  }
  return -1;
}

inline int select_accelerator(const device& dev) {
  if(dev.is_accelerator()) {
    if(dev.hipSYCL_has_compiled_kernels())
      return 2;
    else
      return 1;
  }
  return -1;
}

inline int select_cpu(const device& dev) {
  if(dev.is_cpu())
    return 1;
  return -1;
}

inline int select_host(const device& dev) {
  return select_cpu(dev);
}

inline int select_default(const device& dev) {
#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__) ||                                 \
    defined(__HIPSYCL_ENABLE_HIP_TARGET__) ||                                  \
    defined(__HIPSYCL_ENABLE_SPIRV_TARGET__)
  // Add 2 to make sure that, if no GPU is found
  if(!dev.is_cpu() && dev.hipSYCL_has_compiled_kernels()) {
    // Prefer GPUs (or other accelerators) that have been targeted
    // and have compiled kernels
    return 2;
  } else if(dev.is_cpu()) {
    // Prefer CPU over GPUs that don't have compiled kernels
    // and cannot run kernels.
    return 1;
  } else {
    // Last option: GPUs without compiled kernels
    // This should never be selected in practice because
    // there's always a CPU device.
    return 0;
  }
#else
  return select_host(dev);
#endif
}

template<class Selector>
device select_device(const Selector& s) {
  auto devices = device::get_devices();
  // There should always be at least a CPU device
  assert(devices.size() > 0);

  int best_score = std::numeric_limits<int>::min();
  device candidate;
  for (const device &d : devices) {
    int current_score = s(d);
    if (current_score > best_score) {
      best_score = current_score;
      candidate = d;
    }
  }
  if (best_score < 0) {
    throw sycl::runtime_error{"No matching device"};
  }

  return candidate;
}

template<class T>
struct is_device_selector {
  static constexpr bool value =
      std::is_convertible_v<T, std::function<int(const device &)>>;
};

template<class T>
inline constexpr bool is_device_selector_v = is_device_selector<T>::value;

}

/// Provided only for backwards-compatibility with SYCL 1.2.1
/// so users can still derive custom selectors from device_selector
class device_selector
{
public:
  virtual ~device_selector(){};
  
  device select_device() const {
    return detail::select_device(*this);
  }

  virtual int operator()(const device& dev) const = 0;

};

/// Old SYCL 1.2.1 types are still required for backwards compatibility
/// Note: SYCL 2020 does not mandate how they are implemented - in
/// particular, they don't have to be derived from device_selector!
class error_selector {
public:
  int operator()(const device &dev) const {
    throw unimplemented{"error_selector device selection invoked"};
  }
};

class gpu_selector {
public:
  int operator()(const device &dev) const { return detail::select_gpu(dev); }
};

class accelerator_selector {
public:
  int operator()(const device &dev) const {
    return detail::select_accelerator(dev);
  }
};

class cpu_selector {
public:
  int operator()(const device &dev) const { return detail::select_cpu(dev); }
};

class host_selector {
public:
  int operator()(const device &dev) const { return detail::select_host(dev); }
};

class default_selector {
public:
  int operator()(const device &dev) const {
    return detail::select_default(dev);
  }
};


inline constexpr default_selector default_selector_v;
inline constexpr cpu_selector cpu_selector_v;
inline constexpr gpu_selector gpu_selector_v;
inline constexpr accelerator_selector accelerator_selector_v;

inline auto aspect_selector(const std::vector<aspect> &aspectList,
                            const std::vector<aspect> &denyList = {}) {

  return [=](const device& dev) {
    if(aspectList.empty() && denyList.empty())
      return detail::select_default(dev);

    for(aspect a : aspectList) {
      if(!dev.has(a))
        return -1;
    }
    for(aspect a : denyList) {
      if(dev.has(a))
        return -1;
    }
    return 1;
  };
}

template <typename... aspectListTN>
auto aspect_selector(aspectListTN... aspectList) {
  return [=](const device& dev) {
    if(sizeof...(aspectList) == 0)
      return detail::select_default(dev);

    bool satisfies_all = (dev.has(aspectList) && ...);
    if(satisfies_all)
      return 1;
    return -1;
  };
}

template <aspect... aspectList>
auto aspect_selector() {
  return aspect_selector(aspectList...);
}

inline device::device(const device_selector &deviceSelector) {
  this->_device_id = deviceSelector.select_device()._device_id;
}




}
}

#endif
