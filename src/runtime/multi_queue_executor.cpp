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
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/dag_direct_scheduler.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"

#include <algorithm>
#include <limits>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

std::size_t determine_target_lane(const dag_node_ptr& node,
                                  const node_list_t& nonvirtual_reqs,
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

  common::small_vector<int, 8> synchronization_cost(lane_range.num_lanes);

  for(auto& req : nonvirtual_reqs){
    assert(req);
    assert(req->is_submitted());

    std::size_t lane_id = 0;
    if(executor->find_assigned_lane_index(req, lane_id)) {
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



} // anonymous namespace

multi_queue_executor::multi_queue_executor(
    const backend &b, queue_factory_function queue_factory)
    : _backend{b.get_unique_backend_id()} {
  std::size_t num_devices = b.get_hardware_manager()->get_num_devices();


  _device_data.resize(num_devices);

  for (std::size_t dev = 0; dev < num_devices; ++dev) {

    device_id dev_id = b.get_hardware_manager()->get_device_id(dev);
    hardware_context *hw_context = b.get_hardware_manager()->get_device(dev);

    std::size_t memcpy_concurrency = hw_context->get_max_memcpy_concurrency();
    std::size_t kernel_concurrency = hw_context->get_max_kernel_concurrency();

    for (std::size_t i = 0; i < memcpy_concurrency; ++i) {
      std::unique_ptr<inorder_queue> new_queue = queue_factory(dev_id);
      _managed_queues.push_back(new_queue.get());
      _device_data[dev].executors.push_back(
          std::make_unique<inorder_executor>(std::move(new_queue)));
    }

    _device_data[dev].memcpy_lanes.begin = 0;
    _device_data[dev].memcpy_lanes.num_lanes = memcpy_concurrency;

    for(std::size_t i  = 0; i < kernel_concurrency; ++i) {
      std::unique_ptr<inorder_queue> new_queue = queue_factory(dev_id);
      _managed_queues.push_back(new_queue.get());
      _device_data[dev].executors.push_back(
          std::make_unique<inorder_executor>(std::move(new_queue)));
    }

    _device_data[dev].kernel_lanes.begin = memcpy_concurrency;
    _device_data[dev].kernel_lanes.num_lanes = kernel_concurrency;

    const std::size_t max_statistics_size = application::get_settings()
            .get<setting::mqe_lane_statistics_max_size>();
    const double statistics_decay_time_sec = application::get_settings()
                      .get<setting::mqe_lane_statistics_decay_time_sec>();

    _device_data[dev].submission_statistics = moving_statistics{
        max_statistics_size,
        _device_data[dev].executors.size(),
        static_cast<std::size_t>(1e9 * statistics_decay_time_sec)};
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
  return false;
}

bool multi_queue_executor::is_outoforder_queue() const {
  return true;
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
    const dag_node_ptr& node, operation *op,
    const node_list_t &reqs) {

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
  
  inorder_executor *executor = _device_data[node->get_assigned_device().get_id()]
                         .executors[op_target_lane]
                         .get();

  HIPSYCL_DEBUG_INFO
      << "multi_queue_executor: Dispatching to lane " << op_target_lane << ": "
      << dump(op) << std::endl;
  
  return executor->submit_directly(node, op, reqs);
}

bool multi_queue_executor::can_execute_on_device(const device_id &dev) const {
  return _backend == dev.get_backend();
}

bool multi_queue_executor::is_submitted_by_me(const dag_node_ptr& node) const {
  if(!node->is_submitted())
    return false;

  for(const auto& d : _device_data) {
    for(const auto& executor : d.executors) {
      if(executor->is_submitted_by_me(node))
        return true;
    }
  }

  return false;
}

bool multi_queue_executor::find_assigned_lane_index(
    const dag_node_ptr &node, std::size_t &index_out) const {
  if(!node->is_submitted())
    return false;

  std::size_t dev_id = node->get_assigned_device().get_id();
  std::size_t lane_id = 0;

  for(const auto& executor : _device_data[dev_id].executors) {
    if(executor->is_submitted_by_me(node)) {
      index_out = lane_id;
      return true;
    }

    ++lane_id;
  }

  return false;
}
}
}
