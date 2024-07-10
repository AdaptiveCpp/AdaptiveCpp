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
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/common/debug.hpp"

#include <cassert>
#include <mutex>

namespace hipsycl {
namespace rt {

worker_thread::worker_thread()
    : _continue{true}
{
  _worker_thread = std::thread{[this](){ work(); } };
}


worker_thread::~worker_thread()
{
  halt();

  assert(_enqueued_operations.empty());
}

void worker_thread::wait()
{
  std::unique_lock<std::mutex> lock(_mutex);
  if(!_enqueued_operations.empty())
  {
    // Before going to sleep, wake up the other thread to avoid deadlocks
    _condition_wait.notify_all();
    // Wait until no operation is pending
    _condition_wait.wait(lock, [this]{return _enqueued_operations.empty();});
  }
  assert(_enqueued_operations.empty());
}


void worker_thread::halt()
{
  wait();

  {
    std::unique_lock<std::mutex> lock(_mutex);
    _continue = false;
    _condition_wait.notify_all();
  }
  if(_worker_thread.joinable())
    _worker_thread.join();
}

void worker_thread::work()
{
  // This is the main function executed by the worker thread.
  // The loop is executed as long as there are enqueued operations,
  // (_is_operation_pending) or we should wait for new operations
  // (_continue).
  while(_continue || queue_size() > 0)
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);

      // Before going to sleep, wake up the other thread in case it is is waiting
      // for the queue to get empty
      _condition_wait.notify_all();
      // Wait until we have work, or until _continue becomes false
      _condition_wait.wait(lock,
                           [this](){
        return _enqueued_operations.size()>0 || !_continue;
      });
    }

    // In any way, process the pending operations

    async_function operation = [](){};
    bool has_operation = false;

    {
      std::lock_guard<std::mutex> lock(_mutex);

      if(!_enqueued_operations.empty())
      {
        operation = _enqueued_operations.front();
        has_operation = true;
      }
    }

    operation();

    {
      std::lock_guard<std::mutex> lock{_mutex};
      if(has_operation)
        _enqueued_operations.pop();
    }

    _condition_wait.notify_all();

  }
}

void worker_thread::operator()(worker_thread::async_function f)
{
  std::unique_lock<std::mutex> lock(_mutex);

  _enqueued_operations.push(f);

  lock.unlock();
  _condition_wait.notify_all();
}

std::size_t worker_thread::queue_size() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _enqueued_operations.size();
}


}
}
