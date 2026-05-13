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
#include "hipSYCL/runtime/vk/vk_event.hpp"
#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"
#include "hipSYCL/runtime/vk/vk_queue.hpp"

namespace hipsycl {
namespace rt {

vk_node_event::vk_node_event(vk_queue *queue, uint64_t val)
    : _queue{queue}, _signal_val(val) {}

bool vk_node_event::is_complete() const {
  const uint64_t counter = _queue->get_semaphore_counter_value();
  return counter >= _signal_val;
}

void vk_node_event::wait() {
  vk::Semaphore semaphore = _queue->get_semaphore();

  {
    std::stringstream ss;
    ss << "vk_event: semaphore " << semaphore << " wait on " << _signal_val
       << std::endl;
    HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
  }

  vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &_signal_val);

  vk::Result wait_ret_code;
  do {
    wait_ret_code = _queue->get_dev_ctx()->get_device().waitSemaphores(
        wait_info, UINT64_MAX);
  } while (vk::Result::eTimeout == wait_ret_code);

  if (wait_ret_code != vk::Result::eSuccess) {
    std::string err_msg("Semaphore wait failed with unexpected return code ");
    err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
    print_error(__acpp_here(), error_info{err_msg});
  }
}

vk_node_event::backend_event_type vk_node_event::get_event() const {
  return _queue->get_semaphore();
}

device_id vk_node_event::get_device() const { return _queue->get_device(); }

vk_node_event::backend_event_type vk_node_event::request_backend_event() {
  return get_event();
}

uint64_t vk_node_event::get_signal_val() const { return _signal_val; }

} // namespace rt
} // namespace hipsycl
