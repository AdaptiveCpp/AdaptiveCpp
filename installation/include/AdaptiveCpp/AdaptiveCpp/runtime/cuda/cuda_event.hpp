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
#ifndef HIPSYCL_CUDA_EVENT_HPP
#define HIPSYCL_CUDA_EVENT_HPP

#include "../inorder_queue_event.hpp"

// Note: CUevent_st* == cudaEvent_t 
struct CUevent_st;

namespace hipsycl {
namespace rt {

class cuda_event_pool;
class cuda_node_event : public inorder_queue_event<CUevent_st*>
{
public:
  using backend_event_type = CUevent_st*;
  /// \param evt cuda event; must have been properly initialized and recorded.
  /// \param pool the pool managing the event. If not null, the destructor will return the event
  /// to the pool.
  cuda_node_event(device_id dev, CUevent_st* evt, cuda_event_pool* pool = nullptr);

  ~cuda_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  backend_event_type get_event() const;
  device_id get_device() const;

  backend_event_type request_backend_event() override;
private:
  device_id _dev;
  backend_event_type _evt;
  cuda_event_pool* _pool;
};

}
}


#endif