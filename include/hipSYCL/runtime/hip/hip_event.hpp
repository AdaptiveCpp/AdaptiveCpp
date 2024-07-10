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
#ifndef HIPSYCL_HIP_EVENT_HPP
#define HIPSYCL_HIP_EVENT_HPP

#include "../inorder_queue_event.hpp"

struct ihipEvent_t;

namespace hipsycl {
namespace rt {

class hip_event_pool;
class hip_node_event : public inorder_queue_event<ihipEvent_t*>
{
public:
  using backend_event_type = ihipEvent_t*;
  /// \param evt Must have been properly initialized and recorded.
  /// \param pool the pool managing the event. If not null, the destructor
  /// will return the event to the pool.
  hip_node_event(device_id dev, backend_event_type evt, hip_event_pool* pool = nullptr);

  ~hip_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  ihipEvent_t* get_event() const;
  device_id get_device() const;

  virtual backend_event_type request_backend_event() override;
private:
  device_id _dev;
  backend_event_type _evt;
  hip_event_pool* _pool;
};

}
}


#endif