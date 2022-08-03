/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#include <level_zero/ze_api.h>

#include "hipSYCL/runtime/ze/ze_event.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

ze_node_event::ze_node_event(ze_event_handle_t evt,
                             std::shared_ptr<ze_event_pool_handle_t> pool)
    : _evt{evt}, _pool{pool} {}

ze_node_event::~ze_node_event() {
  ze_result_t err = zeEventDestroy(_evt);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__hipsycl_here(),
                  error_info{"ze_node_event: Could not destroy event",
                             error_code{"ze", static_cast<int>(err)}});
  }
}

bool ze_node_event::is_complete() const {
  ze_result_t err = zeEventQueryStatus(_evt);

  if(err != ZE_RESULT_SUCCESS && err != ZE_RESULT_NOT_READY) {
    register_error(__hipsycl_here(),
                  error_info{"ze_node_event: Could not query event status",
                             error_code{"ze", static_cast<int>(err)}});
  }

  return err == ZE_RESULT_SUCCESS;
}

void ze_node_event::wait() {

  ze_result_t err = zeEventHostSynchronize(_evt, UINT64_MAX);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__hipsycl_here(),
                  error_info{"ze_node_event: Could not wait for event",
                             error_code{"ze", static_cast<int>(err)}});
  }
}

ze_event_handle_t ze_node_event::get_event_handle() const {
  return _evt;
}

ze_event_handle_t ze_node_event::request_backend_event() {
  return get_event_handle();
}

}
}

