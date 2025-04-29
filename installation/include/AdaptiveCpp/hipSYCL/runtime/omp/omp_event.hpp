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
#ifndef HIPSYCL_OMP_EVENT_HPP
#define HIPSYCL_OMP_EVENT_HPP

#include "../inorder_queue_event.hpp"
#include "../signal_channel.hpp"
#include <memory>

namespace hipsycl {
namespace rt {

class omp_node_event
    : public inorder_queue_event<std::shared_ptr<signal_channel>> {
public:
  
  omp_node_event();
  ~omp_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  std::shared_ptr<signal_channel> get_signal_channel() const;

  virtual std::shared_ptr<signal_channel> request_backend_event() override;
private:

  std::shared_ptr<signal_channel> _signal_channel;
};
}
}

#endif
