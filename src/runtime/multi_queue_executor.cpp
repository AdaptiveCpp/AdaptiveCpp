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

#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"

#include <memory>

namespace hipsycl {
namespace rt {

namespace {

class queue_operation_dispatcher : public operation_dispatcher
{
public:
  queue_operation_dispatcher(inorder_queue* q)
  : _queue{q}
  {}

  virtual ~queue_operation_dispatcher(){}

  virtual result dispatch_kernel(kernel_operation* op) final override {
    return _queue->submit_kernel(*op);
  }

  virtual result dispatch_memcpy(memcpy_operation* op) final override {
    return _queue->submit_memcpy(*op);
  }

  virtual result dispatch_prefetch(prefetch_operation* op) final override {
    return _queue->submit_prefetch(*op);
  }

  virtual result dispatch_memset(memset_operation *op) final override {
    return _queue->submit_memset(*op);
  }

private:
  inorder_queue* _queue;
};

} // anonymous namespace

multi_queue_executor::multi_queue_executor(
    const backend &b, queue_factory_function queue_factory) {
  std::size_t num_devices = b.get_hardware_manager()->get_num_devices();


  _device_data.resize(num_devices);

  for (std::size_t dev = 0; dev < num_devices; ++dev) {

    device_id dev_id = b.get_hardware_manager()->get_device_id(dev);
    hardware_context *hw_context = b.get_hardware_manager()->get_device(dev);

    std::size_t memcpy_concurrency = hw_context->get_max_memcpy_concurrency();
    std::size_t kernel_concurrency = hw_context->get_max_kernel_concurrency();

    for (std::size_t i = 0; i < memcpy_concurrency; ++i) {
      _device_data[dev].queues.push_back(queue_factory(dev_id));
    }

    _device_data[dev].memcpy_lanes.begin = 0;
    _device_data[dev].memcpy_lanes.num_lanes = memcpy_concurrency;

    for(std::size_t i  = 0; i < kernel_concurrency; ++i) {
      _device_data[dev].queues.push_back(queue_factory(dev_id));
    }

    _device_data[dev].kernel_lanes.begin = memcpy_concurrency;
    _device_data[dev].kernel_lanes.num_lanes = kernel_concurrency;
  }
}


bool multi_queue_executor::is_inorder_queue() const {
  return true;
}

bool multi_queue_executor::is_outoforder_queue() const {
  return false;
}

bool multi_queue_executor::is_taskgraph() const {
  return false;
}

backend_execution_lane_range
multi_queue_executor::get_memcpy_execution_lane_range(device_id dev) const {
  assert(static_cast<std::size_t>(dev.get_id()) < _device_data.size());

  return this->_device_data[dev.get_id()].memcpy_lanes;
}

backend_execution_lane_range
multi_queue_executor::get_kernel_execution_lane_range(device_id dev) const {
  assert(static_cast<std::size_t>(dev.get_id()) < _device_data.size());

  return this->_device_data[dev.get_id()].kernel_lanes;
}

void multi_queue_executor::submit_directly(
    dag_node_ptr node, operation *op,
    const std::vector<dag_node_ptr> &reqs) {

  HIPSYCL_DEBUG_INFO << "multi_queue_executor: Processing node " << node.get();
  assert(!op->is_requirement());

  if (node->is_submitted())
    return;

  std::size_t op_target_lane;

  if (op->is_data_transfer()) {
    op_target_lane =
        _device_data[node->get_assigned_device().get_id()].memcpy_lanes.begin;
  } else {
    op_target_lane =
        _device_data[node->get_assigned_device().get_id()].kernel_lanes.begin;
  }
  node->assign_to_execution_lane(op_target_lane);

  inorder_queue *q = _device_data[node->get_assigned_device().get_id()]
                         .queues[op_target_lane]
                         .get();

  // Submit synchronization mechanisms
  result res;
  for (auto req : reqs) {
    // The scheduler should not hand us virtual requirements
    assert(!req->is_virtual());
    assert(req->is_submitted());

    if (req->get_assigned_executor() != this) {
      HIPSYCL_DEBUG_INFO
          << " --> Synchronizes with external node: " << req
          << std::endl;
      res = q->submit_external_wait_for(req);
    } else {
      if (req->get_assigned_execution_lane() == op_target_lane) {
        HIPSYCL_DEBUG_INFO
          << " --> (Skipping same-lane synchronization with node: " << req
          << ")" << std::endl;
        // Nothing to synchronize, the requirement was enqueued on the same
        // inorder queue and will therefore be executed before
        // the new node
      } else {
        assert(req->get_event());
        HIPSYCL_DEBUG_INFO << " --> Synchronizes with other queue for node: "
                           << req
                           << " lane = " << req->get_assigned_execution_lane()
                           << std::endl;
        res = q->submit_queue_wait_for(req->get_event());
      }
    }
    if (!res.is_success()) {
      register_error(res);
      node->cancel();
      return;
    }
  }

  HIPSYCL_DEBUG_INFO
      << "multi_queue_executor: Dispatching to lane " << op_target_lane << ": "
      << dump(op) << std::endl;
  
  queue_operation_dispatcher dispatcher{q};
  res = op->dispatch(&dispatcher);
  if (!res.is_success()) {
    register_error(res);
    node->cancel();
    return;
  }

  node->mark_submitted(q->insert_event());
}



}
}
