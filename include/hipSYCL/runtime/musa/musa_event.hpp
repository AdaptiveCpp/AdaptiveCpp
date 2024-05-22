/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_MUSA_EVENT_HPP
#define HIPSYCL_MUSA_EVENT_HPP

#include "../inorder_queue_event.hpp"

// Note: MUevent_st* == musaEvent_t 
struct MUevent_st;

namespace hipsycl {
namespace rt {

class musa_event_pool;
class musa_node_event : public inorder_queue_event<MUevent_st*>
{
public:
  using backend_event_type = MUevent_st*;
  /// \param evt musa event; must have been properly initialized and recorded.
  /// \param pool the pool managing the event. If not null, the destructor will return the event
  /// to the pool.
  musa_node_event(device_id dev, MUevent_st* evt, musa_event_pool* pool = nullptr);

  ~musa_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  backend_event_type get_event() const;
  device_id get_device() const;

  backend_event_type request_backend_event() override;
private:
  device_id _dev;
  backend_event_type _evt;
  musa_event_pool* _pool;
};

}
}


#endif
