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
#ifndef HIPSYCL_SIGNAL_CHANNEL_HPP
#define HIPSYCL_SIGNAL_CHANNEL_HPP

#include <future>
#include <chrono>


namespace hipsycl {
namespace rt {

class signal_channel {
public:
  signal_channel() {
    _shared_future = _promise.get_future().share();
    _has_signalled_flag = false;
  }

  void signal() {
    _has_signalled_flag = true;
    _promise.set_value(true);
  }

  void wait() {
    auto future = _shared_future;
    future.wait();
  }

  bool has_signalled() const {
    return _has_signalled_flag;
  }

private:
  std::promise<bool> _promise;
  std::shared_future<bool> _shared_future;
  // This flag is a workaround for a serious performance
  // bug in libstdc++ prior to version 11, where the
  // future::wait_for(duration(0)) pattern is extremely inefficient
  std::atomic<bool> _has_signalled_flag;
};

}
}

#endif
