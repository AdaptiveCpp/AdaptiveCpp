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
#include <memory>
#include <level_zero/ze_api.h>

#ifndef HIPSYCL_ZE_EVENT_HPP
#define HIPSYCL_ZE_EVENT_HPP

#include "../inorder_queue_event.hpp"

namespace hipsycl {
namespace rt {


class ze_node_event : public inorder_queue_event<ze_event_handle_t>
{
public:
  /// Takes ownership of supplied ze_event_handle_t
  ze_node_event(ze_event_handle_t evt, 
    std::shared_ptr<ze_event_pool_handle_t> pool);
  ~ze_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  ze_event_handle_t get_event_handle() const;
  virtual ze_event_handle_t request_backend_event() override;
private:
  ze_event_handle_t _evt;
  std::shared_ptr<ze_event_pool_handle_t> _pool;
};


}
}

#endif
