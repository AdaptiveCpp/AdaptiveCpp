/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#ifndef HIPSYCL_QUEUE_COMPLETION_EVENT_HPP
#define HIPSYCL_QUEUE_COMPLETION_EVENT_HPP

#include <mutex>
#include "error.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "inorder_queue_event.hpp"
#include "inorder_queue.hpp"


namespace hipsycl {
namespace rt {


template <class FineGrainedBackendEventT, class FineGrainedInorderQueueEventT>
class queue_completion_event
    : public inorder_queue_event<FineGrainedBackendEventT> {
public:
  queue_completion_event(inorder_queue* q)
  : _q{q}, _has_fine_grained_event{false}, _is_complete{false} {}

  virtual ~queue_completion_event(){}

  virtual bool is_complete() const override {
    if(_is_complete)
      return true;
    
    if(_has_fine_grained_event)
      return _fine_grained_event->is_complete();
    
    inorder_queue_status status;
    auto err = _q->query_status(status);
    if(!err.is_success()) {
      register_error(err);
    }

    return status.is_complete();
  }

  virtual void wait() override {
    if(_is_complete)
      return;

    if(_has_fine_grained_event)
      _fine_grained_event->wait();
    else
      _q->wait();
    _is_complete = true;
  }

  virtual FineGrainedBackendEventT request_backend_event() override {
    if(!_has_fine_grained_event){
      // Avoid another thread from also entering this code path -
      // setting _has_fined_grained_event to true and setting
      // _fine_grained_event must happend atomically.
      std::lock_guard<std::mutex> lock{_decay_to_fine_grained_evt_mutex};
      if(!_has_fine_grained_event) {
        // We need to first create an actual fine-grained event
        // to be able to service the request, since this event
        // is not tied to a backend event otherwise.
        _fine_grained_event = _q->insert_event();
        _has_fine_grained_event = true;
      }
    }
    
    return static_cast<inorder_queue_event<FineGrainedBackendEventT> *>(
               _fine_grained_event.get())
        ->request_backend_event();
  }
private:
  inorder_queue* _q;
  std::atomic<bool> _has_fine_grained_event;
  std::atomic<bool> _is_complete;
  std::shared_ptr<dag_node_event> _fine_grained_event;
  std::mutex _decay_to_fine_grained_evt_mutex;
};

}
}

#endif
