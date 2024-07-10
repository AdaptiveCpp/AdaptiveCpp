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
    register_error(__acpp_here(),
                  error_info{"ze_node_event: Could not destroy event",
                             error_code{"ze", static_cast<int>(err)}});
  }
}

bool ze_node_event::is_complete() const {
  ze_result_t err = zeEventQueryStatus(_evt);

  if(err != ZE_RESULT_SUCCESS && err != ZE_RESULT_NOT_READY) {
    register_error(__acpp_here(),
                  error_info{"ze_node_event: Could not query event status",
                             error_code{"ze", static_cast<int>(err)}});
  }

  return err == ZE_RESULT_SUCCESS;
}

void ze_node_event::wait() {

  ze_result_t err = zeEventHostSynchronize(_evt, UINT64_MAX);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(__acpp_here(),
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

