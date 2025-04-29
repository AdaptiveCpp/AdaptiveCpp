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
#ifndef HIPSYCL_HIP_DEVICE_MANAGER_HPP
#define HIPSYCL_HIP_DEVICE_MANAGER_HPP

namespace hipsycl {
namespace rt {

/// HIP keeps track of the currently active device on a per-thread basis.
/// The hip_device_manager acts as a wrapper for this functionality.
/// It is implemented as a per-thread singleton and assumes that
/// no external calls to hipSetDevice() are made by the user.
class hip_device_manager
{
public:
  void activate_device(int device_id);
  int get_active_device() const;

  static hip_device_manager &get() {
    static thread_local hip_device_manager instance;
    return instance;
  }

private:
  int _device;

  hip_device_manager();
};

}
}

#endif
