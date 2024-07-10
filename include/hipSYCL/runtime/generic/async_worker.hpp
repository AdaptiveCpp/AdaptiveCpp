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

  std::atomic<bool> _continue;

  std::condition_variable _condition_wait;
  mutable std::mutex _mutex;

  std::queue<async_function> _enqueued_operations;
};

}
}

#endif
