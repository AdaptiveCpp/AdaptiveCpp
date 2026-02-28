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
#ifndef HIPSYCL_METAL_EVENT_HPP
#define HIPSYCL_METAL_EVENT_HPP

#include "../inorder_queue_event.hpp"
#include "../signal_channel.hpp"
#include <memory>

namespace MTL {
class SharedEvent;
} // namespace MTL

namespace hipsycl {
namespace rt {

struct metal_event_handle {
  MTL::SharedEvent* event;
  uint64_t value;
};

class metal_node_event
  : public inorder_queue_event<metal_event_handle> {
public:
  metal_node_event() = delete;
  metal_node_event(metal_event_handle handle);
  ~metal_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  virtual metal_event_handle request_backend_event() override;

private:
  metal_event_handle _handle;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_EVENT_HPP
