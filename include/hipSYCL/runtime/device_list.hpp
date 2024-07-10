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
#ifndef HIPSYCL_RT_DEVICE_LIST_HPP
#define HIPSYCL_RT_DEVICE_LIST_HPP

#include <algorithm>
#include <vector>

#include "runtime.hpp"
#include "device_id.hpp"

namespace hipsycl {
namespace rt {

class unique_device_list {
public:
  unique_device_list(runtime* rt)
  : _rt{rt} {}

  void add(const rt::device_id dev) {
    if (std::find(_devices.begin(), _devices.end(), dev) == _devices.end()) {
      _devices.push_back((dev));
      if (std::find(_backends.begin(), _backends.end(), dev.get_backend()) ==
          _backends.end()) {
        _backends.push_back(dev.get_backend());
      }
    }
  }

  void add(const unique_device_list &other) {
    other.for_each_device([this](rt::device_id dev) {
      add(dev);
    });
  }

  template <class F> void for_each_device(F f) const {
    for (const device_id &dev : _devices) {
      f(dev);
    }
  }

  template <class F> void for_each_backend(F f) const {
    for (const backend_id &b : _backends) {
      f(b);
    }
  }

  friend bool operator==(const unique_device_list &a,
                         const unique_device_list &b) {
    return a._devices == b._devices;
  }

  friend bool operator!=(const unique_device_list &a,
                         const unique_device_list &b) {
    return !(a == b);
  }

  std::size_t get_num_backends(hardware_platform plat) const {
    std::size_t count = 0;

    for_each_backend([&](backend_id b) {
      if (_rt->backends().get(b)->get_hardware_platform() == plat)
        ++count;
    });
    
    return count;
  }

  std::size_t get_num_backends() const { return _backends.size(); }

  using backend_iterator = std::vector<backend_id>::const_iterator;
  using device_iterator = std::vector<device_id>::const_iterator;

  backend_iterator backends_begin() const { return _backends.begin(); }
  backend_iterator backends_end() const { return _backends.end(); }
  device_iterator devices_begin() const { return _devices.begin(); }
  device_iterator devices_end() const { return _devices.end(); }

  template <class UnaryPredicate>
  device_iterator find_first_device(UnaryPredicate p) const {
    return std::find_if(devices_begin(), devices_end(), p);
  }

  template <class UnaryPredicate>
  backend_iterator find_first_backend(UnaryPredicate p) const {
    return std::find_if(backends_begin(), backends_end(), p);
  }

  bool contains_device(const device_id& dev) const {
    return std::find(_devices.begin(), _devices.end(), dev) != _devices.end();
  }

  runtime* get_runtime() const {
    return _rt;
  }
private:
  std::vector<device_id> _devices;
  std::vector<backend_id> _backends;
  runtime* _rt;
};

}
} // namespace hipsycl

#endif
