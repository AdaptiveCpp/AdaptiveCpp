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

#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

void hip_event_deleter::operator()(hipEvent_t evt) const {
  auto err = hipEventDestroy(evt);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_node_event: Couldn't destroy event",
                              error_code{"HIP", err}});
  }
}

hip_unique_event make_hip_event() {
  hipEvent_t evt;
  if (hipError_t err = hipEventCreate(&evt); err != hipSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"hip_event: Couldn't create event", error_code{"HIP", err}});
    return nullptr;
  } else {
    return hip_unique_event{evt};
  }
}


hip_node_event::hip_node_event(device_id dev, hip_unique_event evt)
: _dev{dev}, _evt{std::move(evt)}
{}

bool hip_node_event::is_complete() const
{
  hipError_t err = hipEventQuery(_evt.get());
  if (err != hipErrorNotReady && err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_node_event: Couldn't query event status",
                              error_code{"HIP", err}});
  }
  return err == hipSuccess;
}

void hip_node_event::wait()
{
  auto err = hipEventSynchronize(_evt.get());
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_node_event: hipEventSynchronize() failed",
                              error_code{"HIP", err}});
  }
}

hipEvent_t hip_node_event::get_event() const
{
  return _evt.get();
}

device_id hip_node_event::get_device() const
{
  return _dev;
}

}
}
