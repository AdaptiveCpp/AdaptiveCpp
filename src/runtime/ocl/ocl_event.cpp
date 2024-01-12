/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
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

#include "hipSYCL/runtime/ocl/ocl_event.hpp"
#include "hipSYCL/runtime/device_id.hpp"
namespace hipsycl {
namespace rt {

ocl_node_event::ocl_node_event(device_id dev, cl::Event evt)
: _dev{dev}, _evt{evt} {}

ocl_node_event::ocl_node_event(device_id dev)
: _is_empty{true}, _dev{dev}, _evt{} {}

ocl_node_event::~ocl_node_event(){}

bool ocl_node_event::is_complete() const {
  if(_is_empty)
    return true;

  cl_int status;
  cl_int err = _evt.getInfo(CL_EVENT_COMMAND_EXECUTION_STATUS, &status);

  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_node_event: Couldn't query event status",
                              error_code{"CL", err}});
    return false;
  }

  if(status < 0) {
    // Command was abnormally terminated
    register_error(__hipsycl_here(),
                   error_info{"ocl_node_event: Event status indicates that "
                              "device operation was abnormally terminated",
                              error_code{"CL", err}});
    return true;
  }
  
  return status == CL_COMPLETE;
}

void ocl_node_event::wait() {
  if(_is_empty)
    return;

  cl_int err = _evt.wait();
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_node_event: Waiting for event failed",
                              error_code{"CL", err}});
  }
}

ocl_node_event::backend_event_type ocl_node_event::get_event() const {
  return _evt;
}

device_id ocl_node_event::get_device() const {
  return _dev;
}

ocl_node_event::backend_event_type ocl_node_event::request_backend_event() {
  return get_event();
}



}
}
