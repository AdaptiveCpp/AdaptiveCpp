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

#ifndef SYCU_ASYNC_WORKER_HPP
#define SYCU_ASYNC_WORKER_HPP

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <functional>


namespace cl {
namespace sycl {
namespace detail {


/// A worker thread that executes exactly one task in the background.
/// If a second task is enqueued, waits until the first task
/// has completed.
class worker_thread
{
public:
  /// Construct object
  worker_thread();


  worker_thread(const worker_thread&) = delete;
  worker_thread& operator=(const worker_thread&) = delete;

  ~worker_thread();

  /// If a task is currently running, waits until it
  /// has completed.
  void wait();

  /// Enqueues a user-specified function for asynchronous
  /// execution in the worker thread. If another task is
  /// still pending, waits until this task has completed.
  /// \tparam Function A callable object with signature void(void).
  /// \param f The function to enqueue for execution
  template<class Function>
  void operator()(Function f)
  {
    wait();

    std::unique_lock<std::mutex> lock(_mutex);
    _async_operation = f;
    _is_operation_pending = true;

    lock.unlock();
    _condition_wait.notify_one();
  }

  /// \return whether there is currently an operation
  /// pending.
  inline
  bool is_currently_working() const;
private:

  /// Stop the worker thread - this should only be
  /// done in the destructor.
  void halt();

  /// Starts the worker thread, which will execute the supplied
  /// tasks. If no tasks are available, waits until a new task is
  /// supplied.
  void work();

  bool _is_operation_pending;
  std::thread _worker_thread;

  bool _continue;

  std::condition_variable _condition_wait;
  std::mutex _mutex;

  std::function<void()> _async_operation;
};

}
}
}

#endif
