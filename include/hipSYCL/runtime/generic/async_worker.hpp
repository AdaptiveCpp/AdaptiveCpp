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

#ifndef HIPSYCL_ASYNC_WORKER_HPP
#define HIPSYCL_ASYNC_WORKER_HPP

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>
#include <queue>


namespace hipsycl {
namespace rt {

/// A worker thread that processes a queue in the background.
class worker_thread
{
public:
  using async_function = std::function<void ()>;

  /// Construct object
  worker_thread();

  worker_thread(const worker_thread&) = delete;
  worker_thread& operator=(const worker_thread&) = delete;

  ~worker_thread();

  /// Waits until all enqueued tasks have completed.
  void wait();

  /// Enqueues a user-specified function for asynchronous
  /// execution in the worker thread.
  /// \param f The function to enqueue for execution
  void operator()(async_function f);

  /// \return The number of enqueued operations
  std::size_t queue_size() const;

  /// Stop the worker thread
  void halt();
private:

  /// Starts the worker thread, which will execute the supplied
  /// tasks. If no tasks are available, waits until a new task is
  /// supplied.
  void work();

  std::thread _worker_thread;

  bool _continue;

  std::condition_variable _condition_wait;
  mutable std::mutex _mutex;

  std::queue<async_function> _enqueued_operations;
};

}
}

#endif
