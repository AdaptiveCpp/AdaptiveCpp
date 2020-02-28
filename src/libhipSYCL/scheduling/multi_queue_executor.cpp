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

#include "CL/sycl/detail/scheduling/multi_queue_executor.hpp"
#include "CL/sycl/detail/scheduling/dag_interpreter.hpp"
#include <memory>

namespace cl {
namespace sycl {
namespace detail {

namespace {



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
  assert(dev.get_id() < _device_data.size());

  return this->_device_data[dev.get_id()].memcpy_lanes;
}

backend_execution_lane_range
multi_queue_executor::get_kernel_execution_lane_range(device_id dev) const {
  assert(dev.get_id() < _device_data.size());

  return this->_device_data[dev.get_id()].kernel_lanes;
}

void multi_queue_executor::submit_dag(
    const dag_interpreter &interpreter, const dag_enumerator &enumerator,
    const std::vector<node_scheduling_annotation> &annotations){

}

std::unique_ptr<event_before_node>
multi_queue_executor::create_event_before(dag_node_ptr node) const {}

std::unique_ptr<event_before_node>
multi_queue_executor::create_event_after(dag_node_ptr node) const {}

// The create_wait_* functions will be called by the scheduler to mark
// synchronization points
std::unique_ptr<wait_for_node_on_same_lane>
multi_queue_executor::create_wait_for_node_same_lane(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {}

std::unique_ptr<wait_for_node_on_same_backend>
multi_queue_executor::create_wait_for_node_same_backend(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {}

std::unique_ptr<wait_for_external_node>
multi_queue_executor::create_wait_for_external_node(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {}

void multi_queue_executor::submit_node(
    dag_node_ptr node, const dag_interpreter &interpreter,
    const std::vector<node_scheduling_annotation> &annotations) {

  interpreter.for_each_requirement(node, [=](dag_node_ptr req){
    if(!req->is_submitted() && req->get_assigned_executor() == this) {
      this->submit_node(req, interpreter, annotations);
    }
  });

  // Obtain queue
  auto dev = node->get_assigned_device();

  std::size_t execution_lane = node->get_assigned_execution_lane();
  assert(execution_lane < _device_data[dev.get_id()].queues.size());
  
  inorder_queue* q = _device_data[dev.get_id()].queues[execution_lane].get();

  std::size_t node_id = node->get_node_id();

  event_before_node* pre_event = annotations[node_id].get_event_before();
  event_after_node* post_event = annotations[node_id].get_event_after();
  // Submit synchronization, if required
  for(auto& sync_op : annotations[node_id].get_synchronization_ops()) {
    //sync_op->
  }

  // Submit pre-event, if required
  if(pre_event)
    pre_event->assign_event(q->insert_event());

  // Submit actual operation
  operation* op = node->get_operation();

  // Submit post-event, if required
  if(post_event)
    post_event->assign_event(q->insert_event());
  
  assert(!op->is_requirement());
}


}
}
}
