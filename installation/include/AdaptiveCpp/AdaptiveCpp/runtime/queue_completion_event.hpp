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
