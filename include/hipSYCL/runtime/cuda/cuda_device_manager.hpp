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
#ifndef HIPSYCL_CUDA_DEVICE_MANAGER_HPP
#define HIPSYCL_CUDA_DEVICE_MANAGER_HPP

namespace hipsycl {
namespace rt {

/// CUDA keeps track of the currently active device on a per-thread basis.
/// The cuda_device_manager acts as a wrapper for this functionality.
/// It is implemented as a per-thread singleton and assumes that
/// no external calls to cudaSetDevice() are made by the user.
class cuda_device_manager
{
public:
  void activate_device(int device_id);
  int get_active_device() const;

  static cuda_device_manager &get() {
    static thread_local cuda_device_manager instance;
    return instance;
  }

private:
  int _device;

  cuda_device_manager();
};

}
}

#endif
