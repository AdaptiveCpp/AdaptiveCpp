/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP
#define HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP

#include <cassert>
#include <functional>

#include "backend.hpp"
#include "device_id.hpp"
#include "executor.hpp"
#include "inorder_queue.hpp"
#include "generic/multi_event.hpp"

namespace hipsycl {
namespace rt {

/// An executor that submits tasks by serializing them onto 
/// to multiple inorder queues (e.g. CUDA streams)
class multi_queue_executor : public backend_executor
{
public:
  using queue_factory_function =
      std::function<std::unique_ptr<inorder_queue>(device_id)>;

  multi_queue_executor(
      const backend& b,
      queue_factory_function queue_factory);

  virtual ~multi_queue_executor() {}

  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;

  // The range of lanes to use for the given device
  backend_execution_lane_range
  get_memcpy_execution_lane_range(device_id dev) const override;

  // The range of lanes to use for the given device
  backend_execution_lane_range
  get_kernel_execution_lane_range(device_id dev) const override;

  virtual void
  submit_directly(dag_node_ptr node, operation *op,
                  const std::vector<dag_node_ptr> &reqs) override;

  template <class F> void for_each_queue(rt::device_id dev, F handler) const {
    assert(dev.get_id() < _device_data.size());
    for (std::size_t i = 0; i < _device_data[dev.get_id()].queues.size(); ++i)
      handler(_device_data[dev.get_id()].queues[i].get());
  }
private:
  using final_nodes_map = std::unordered_map<inorder_queue*, dag_node_ptr>;

  struct per_device_data
  {
    backend_execution_lane_range memcpy_lanes;
    backend_execution_lane_range kernel_lanes;
    std::vector<std::unique_ptr<inorder_queue>> queues;
  };

  std::vector<per_device_data> _device_data;
};

}
}


#endif

