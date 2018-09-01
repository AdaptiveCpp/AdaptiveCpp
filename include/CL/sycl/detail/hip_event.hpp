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

#ifndef HIPSYCL_HIP_EVENT_HPP
#define HIPSYCL_HIP_EVENT_HPP

#include "../backend/backend.hpp"
#include "../exception.hpp"

#include <cassert>

namespace cl {
namespace sycl {
namespace detail {

class hip_event_manager
{
public:
  hip_event_manager()
  {
    detail::check_error(hipEventCreateWithFlags(&_evt,
                                                hipEventDisableTiming));
  }

  ~hip_event_manager()
  {
    detail::check_error(hipEventDestroy(_evt));
  }

  hipEvent_t& get_event()
  {
    return _evt;
  }
private:
  hipEvent_t _evt;
};

using hip_event_ptr = shared_ptr_class<hip_event_manager>;


class hip_event
{
public:
  hip_event()
    : _is_null_event{true}
  {}

  hip_event(const detail::hip_event_ptr& evt)
    : _is_null_event{false}, _evt{evt}
  {}

  void wait() const
  {
    this->wait_until_done();
  }

  bool is_done() const
  {
    if(_is_null_event)
      return true;

    hipError_t err = hipEventQuery(_evt->get_event());
    detail::check_error(err);

    return err == hipSuccess;
  }

  bool operator ==(const hip_event& rhs) const
  { return _evt == rhs._evt; }

  bool operator !=(const hip_event& rhs) const
  { return !(*this == rhs); }

  bool is_null_event() const
  { return _is_null_event; }
private:
  void wait_until_done() const
  {
    if(!_is_null_event)
      detail::check_error(hipEventSynchronize(_evt->get_event()));
  }

  bool _is_null_event;
  detail::hip_event_ptr _evt;
};


/// Inserts an event into the current stream
static hip_event insert_event(hipStream_t stream)
{
  hip_event_ptr evt{new hip_event_manager()};
  detail::check_error(hipEventRecord(evt->get_event(), stream));
  hip_event result{evt};

  assert(!result.is_null_event());
  return result;
}


}
}
}

#endif
