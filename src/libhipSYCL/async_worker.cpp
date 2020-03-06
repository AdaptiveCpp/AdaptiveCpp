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

#include "hipSYCL/sycl/detail/async_worker.hpp"
#include "hipSYCL/sycl/detail/debug.hpp"

#include <cassert>

namespace hipsycl {
namespace sycl {
namespace detail {


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
    _condition_wait.notify_one();
    // Wait until no operation is pending
    _condition_wait.wait(lock, [this]{return _enqueued_operations.empty();});
  }
}


void worker_thread::halt()
{
  wait();

  _continue = false;
  _condition_wait.notify_one();

  if(_worker_thread.joinable())
    _worker_thread.join();
}

void worker_thread::work()
{
  // This is the main function executed by the worker thread.
  // The loop is executed as long as there are enqueued operations,
  // (_is_operation_pending) or we should wait for new operations
  // (_continue).
  while(_continue || _enqueued_operations.size() > 0)
  {
    {
      std::unique_lock<std::mutex> lock(_mutex);

      // Before going to sleep, wake up the other thread in case it is is waiting
      // for the queue to get empty
      _condition_wait.notify_one();
      // Wait until we have work, or until _continue becomes false
      _condition_wait.wait(lock,
                           [this](){
        return _enqueued_operations.size()>0 || !_continue;
      });
    }

    // In any way, process the pending operations

    async_function operation = [](){};

    {
      std::lock_guard<std::mutex> lock(_mutex);

      if(!_enqueued_operations.empty())
      {
        operation = _enqueued_operations.front();
        _enqueued_operations.pop();

        HIPSYCL_DEBUG_INFO << "Async worker thread: Processing op, "
                              "remaining queue size: "
                  << _enqueued_operations.size()
                  << std::endl;
      }
    }

    operation();

    _condition_wait.notify_one();

  }
}

void worker_thread::operator()(worker_thread::async_function f)
{
  std::unique_lock<std::mutex> lock(_mutex);

  _enqueued_operations.push(f);

  lock.unlock();
  _condition_wait.notify_one();
}

std::size_t worker_thread::queue_size() const
{
  std::lock_guard<std::mutex> lock(_mutex);
  return _enqueued_operations.size();
}

}
}
}
