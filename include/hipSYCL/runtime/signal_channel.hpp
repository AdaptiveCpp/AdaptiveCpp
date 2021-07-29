/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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
