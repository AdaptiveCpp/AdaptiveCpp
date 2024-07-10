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
#ifndef HIPSYCL_COMMON_SPIN_LOCK_HPP
#define HIPSYCL_COMMON_SPIN_LOCK_HPP

#include <atomic>

namespace hipsycl {
namespace common {


class spin_lock {
public:
  void lock() {
    while (_lock.test_and_set(std::memory_order_acquire));
  }
  void unlock() {
    _lock.clear(std::memory_order_release);
  }
private:
  std::atomic_flag _lock = ATOMIC_FLAG_INIT;
};

class spin_lock_guard {
public:
  spin_lock_guard(spin_lock& lock) : _lock(lock) {
    _lock.lock();
  }
  ~spin_lock_guard() {
    _lock.unlock();
  }
private:
  spin_lock& _lock;
};



}
}

#endif

