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

#ifndef ACPP_RT_PCUDA_EVENT_HPP
#define ACPP_RT_PCUDA_EVENT_HPP

#include <memory>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"


namespace hipsycl::rt::pcuda {

class pcuda_runtime;

class event {
public:
  static pcudaError_t create(event *&out, pcuda_runtime *,
                             unsigned int flags);
  static pcudaError_t destroy(event* s);
  
  pcudaError_t wait();
  
  bool is_recorded() const;
  bool is_complete() const;

  pcudaError_t record(inorder_queue* q);

  dag_node_event* get_event() const;
  static dag_node_event* get_event(pcudaEvent_t evt);

  std::shared_ptr<dag_node_event> get_event_shared_ptr() const { return _evt; }
  device_id get_device() const { return _dev; }

private:
  std::shared_ptr<dag_node_event> _evt;
  pcuda_runtime* _rt;

  device_id _dev;
};

}

#endif
