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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"

#include <algorithm>
#include <limits>
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


std::size_t determine_target_lane(dag_node_ptr node,
                                  const std::vector<dag_node_ptr>& nonvirtual_reqs,
                                  const multi_queue_executor* executor,
                                  const moving_statistics& device_submission_statistics,
                                  backend_execution_lane_range lane_range) {
  if(lane_range.num_lanes <= 1) {
    return lane_range.begin;
  }

  if(node->get_execution_hints().has_hint<hints::prefer_execution_lane>()) {
    std::size_t preferred_lane = node->get_execution_hints()
                                     .get_hint<hints::prefer_execution_lane>()
                                     ->get_lane_id();
    return lane_range.begin + preferred_lane % lane_range.num_lanes;
  }

  std::vector<int> synchronization_cost(lane_range.num_lanes);

  for(dag_node_ptr req : nonvirtual_reqs){
    assert(req);
    assert(req->is_submitted());

    if(req->get_assigned_executor() == executor) {
      std::size_t lane_id = req->get_assigned_execution_lane();
      if (lane_id >= lane_range.begin &&
          lane_id < lane_range.begin + lane_range.num_lanes) {
        
        std::size_t relative_lane_id = lane_id - lane_range.begin;
        // Don't consider the event if we already know that it is complete
        if(!req->is_known_complete()) {
          ++synchronization_cost[relative_lane_id];
        }
      }
    }
  }
  // Select the lane that would have the *highest* synchronization cost,
  // because by scheduling to this lane all synchronization becomes noops!
  // If there are multiple lanes with same synchronization cost,
  // use the one with lower recent utilization
  auto lane_usage = device_submission_statistics.build_decaying_bins();
  int max_sync_cost = 0;
  double min_usage = std::numeric_limits<double>::max();
  std::size_t current_best_lane = lane_range.begin;

  for (std::size_t i = lane_range.begin;
       i < lane_range.begin + lane_range.num_lanes; ++i) {

    int sync_cost = synchronization_cost[i-lane_range.begin];

    if(sync_cost > max_sync_cost) {
      max_sync_cost = synchronization_cost[i-lane_range.begin];
      current_best_lane = i;
      min_usage = lane_usage[i];
    } else if(sync_cost == max_sync_cost) {
      if(lane_usage[i] < min_usage) {
        min_usage = lane_usage[i];
        current_best_lane = i;
      }
    }
  }

  return current_best_lane;
}

std::size_t
get_maximum_execution_index_for_lane(const std::vector<dag_node_ptr> &nodes,
                                     multi_queue_executor *executor,
                                     std::size_t lane) {
  std::size_t index = 0;
  for (const auto &node : nodes) {
    if (node->is_submitted() && node->get_assigned_executor() == executor &&
        node->get_assigned_execution_lane() == lane) {
      if(node->get_assigned_execution_index() > index)
        index = node->get_assigned_execution_index();
    }
  }
  return index;
}

} // anonymous namespace

multi_queue_executor::multi_queue_executor(
    const backend &b, queue_factory_function queue_factory)
    : _num_submitted_operations{0} {
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
    _device_data[dev].submission_statistics = moving_statistics{
        100, // Store information about last 100 operations
        _device_data[dev].queues.size(),
        10 * static_cast<std::size_t>(1e9) // Forget after 10 seconds
    };
  }

  HIPSYCL_DEBUG_INFO << "multi_queue_executor: Spawned for backend "
                     << b.get_name() << " with configuration: " << std::endl;
  for(std::size_t i = 0; i < _device_data.size(); ++i) {
    HIPSYCL_DEBUG_INFO << "  device " << i << ": "<< std::endl;

    for(std::size_t j = 0; j < _device_data[i].memcpy_lanes.num_lanes; ++j){
      std::size_t lane = j + _device_data[i].memcpy_lanes.begin;
      HIPSYCL_DEBUG_INFO << "    memcpy lane: " << lane << std::endl;
    }
    for(std::size_t j = 0; j < _device_data[i].kernel_lanes.num_lanes; ++j){
      std::size_t lane = j + _device_data[i].kernel_lanes.begin;
      HIPSYCL_DEBUG_INFO << "    kernel lane: " << lane << std::endl;
    }
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

  HIPSYCL_DEBUG_INFO << "multi_queue_executor: Processing node " << node.get()
	  << " with " << reqs.size() << " non-virtual requirement(s) and "
	  << node->get_requirements().size() << " direct requirement(s)." << std::endl;

  assert(!op->is_requirement());

  if (node->is_submitted())
    return;

  std::size_t op_target_lane;

  if (op->is_data_transfer()) {
    op_target_lane = determine_target_lane(
        node, reqs, this,
        _device_data[node->get_assigned_device().get_id()].submission_statistics,
        _device_data[node->get_assigned_device().get_id()].memcpy_lanes);
  } else {
    op_target_lane = determine_target_lane(
        node, reqs, this,
        _device_data[node->get_assigned_device().get_id()].submission_statistics,
        _device_data[node->get_assigned_device().get_id()].kernel_lanes);
  }
  _device_data[node->get_assigned_device().get_id()]
      .submission_statistics.insert(op_target_lane);
  
  node->assign_to_execution_lane(op_target_lane);
  node->assign_execution_index(_num_submitted_operations);
  ++_num_submitted_operations;

  inorder_queue *q = _device_data[node->get_assigned_device().get_id()]
                         .queues[op_target_lane]
                         .get();

  // Submit synchronization mechanisms

  result res;
  for (auto req : reqs) {
    // The scheduler should not hand us virtual requirements
    assert(!req->is_virtual());
    assert(req->is_submitted());

    // Nothing to do if we have to synchronize with
    // an operation that is already known to have completed
    if(!req->is_known_complete()) {
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

          std::size_t lane = req->get_assigned_execution_lane();

          HIPSYCL_DEBUG_INFO << " --> Synchronizes with other queue for node: "
                            << req
                            << " lane = " << lane
                            << std::endl;
          // We only need to actually synchronize with the lane if this req
          // is the operation that has been submitted *last* to the lane
          // out of all requirements in reqs.
          // (Follows from execution lanes being in-order queues)
          //
          // Find the maximum execution index out of all our requirements.
          // Since the execution index is incremented after each submission,
          // this allows us to identify the requirement that was submitted last.
          std::size_t maximum_execution_index =
              get_maximum_execution_index_for_lane(reqs, this, lane);
          
          if(req->get_assigned_execution_index() != maximum_execution_index) {
            HIPSYCL_DEBUG_INFO
                << "  --> (Skipping unnecessary synchronization; another "
                   "requirement follows in the same inorder queue)"
                << std::endl;
          } else {
            res = q->submit_queue_wait_for(req->get_event());
          }
        }
      }
      if (!res.is_success()) {
        register_error(res);
        node->cancel();
        return;
      }
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
