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
#include "hipSYCL/runtime/dag_interpreter.hpp"
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

  std::shared_ptr<dag_node_event> multi_event =
      std::make_shared<dag_multi_node_event>(
          std::vector<std::shared_ptr<dag_node_event>>{});

  final_nodes_map final_nodes;
  dag_multi_node_event *batch_event =
      static_cast<dag_multi_node_event *>(multi_event.get());
  
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    this->submit_node(node, interpreter, annotations, multi_event, final_nodes);
  });

  for(auto final_node : final_nodes){
    inorder_queue* q = final_node.first;
    dag_node_ptr node = final_node.second;

    if(annotations[node->get_node_id()].has_event_after()){
      batch_event->add_event(node->get_event());
    } else {
      batch_event->add_event(q->insert_event());
    }
  }

}

std::unique_ptr<event_before_node>
multi_queue_executor::create_event_before(dag_node_ptr node) const {
  return std::make_unique<event_before_node>();
}

std::unique_ptr<event_after_node>
multi_queue_executor::create_event_after(dag_node_ptr node) const {
  return std::make_unique<event_after_node>();
}

// The create_wait_* functions will be called by the scheduler to mark
// synchronization points
std::unique_ptr<wait_for_node_on_same_lane>
multi_queue_executor::create_wait_for_node_same_lane(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {

  return std::make_unique<wait_for_node_on_same_lane>(other);
}

std::unique_ptr<wait_for_node_on_same_backend>
multi_queue_executor::create_wait_for_node_same_backend(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {

  other_annotation.insert_event_after_if_missing(
      other->get_assigned_executor()->create_event_after(other));
  
  return std::make_unique<wait_for_node_on_same_backend>(other);

}

std::unique_ptr<wait_for_external_node>
multi_queue_executor::create_wait_for_external_node(
    dag_node_ptr node, const node_scheduling_annotation &annotation,
    dag_node_ptr other,
    node_scheduling_annotation &other_annotation) const {

  other_annotation.insert_event_after_if_missing(
      other->get_assigned_executor()->create_event_after(other));

  return std::make_unique<wait_for_external_node>(other);
}

void multi_queue_executor::submit_node(
    dag_node_ptr node, const dag_interpreter &interpreter,
    const std::vector<node_scheduling_annotation> &annotations,
    std::shared_ptr<dag_node_event> generic_batch_event,
    final_nodes_map& final_nodes) {

  operation* op = node->get_operation();
  assert(!op->is_requirement());

  if(node->is_submitted() || node->get_assigned_executor() != this)
    return;

  interpreter.for_each_requirement(node, [&](dag_node_ptr req){
    if(!req->is_submitted() && req->get_assigned_executor() == this) {
      this->submit_node(req, interpreter, annotations, generic_batch_event,
                        final_nodes);
    }
  });

  // Obtain queue
  auto dev = node->get_assigned_device();

  std::size_t execution_lane = node->get_assigned_execution_lane();
  assert(execution_lane < _device_data[dev.get_id()].queues.size());
  
  inorder_queue* q = _device_data[dev.get_id()].queues[execution_lane].get();

  std::size_t node_id = node->get_node_id();
  queue_operation_dispatcher dispatcher{q};

  event_before_node* pre_event = annotations[node_id].get_event_before();
  event_after_node* post_event = annotations[node_id].get_event_after();
  // Submit synchronization, if required
  for(auto& sync_op : annotations[node_id].get_synchronization_ops()) {
    if(sync_op->is_wait_operation()){
      wait_operation* op = cast<wait_operation>(sync_op.get());

      if(op->get_wait_target() == wait_target::same_lane) {
        // No need to explicitly wait for tasks on the same lane of an inorder
        // queue

        // Since this node is a dependency of this one, it should have already
        // been submitted.
        assert(op->get_target_node()->is_submitted());
      }
      else if(op->get_wait_target() == wait_target::same_backend) {
        // Same backend, but different lane

        // Since this node is a dependency of this one, it should have already
        // been submitted.
        assert(op->get_target_node()->is_submitted());
        // Check that the target node has an event right after it.
        // Otherwise, if we use the generic batch-end event, we would
        // create a deadlock here.
        std::size_t target_node_id = op->get_target_node()->get_node_id();
        assert(annotations[target_node_id].has_event_after());
        
        q->submit_queue_wait_for(op->get_target_node()->get_event());
      }
      else if(op->get_wait_target() == wait_target::external_backend) {
        // Since this node is a dependency of this one, it should have already
        // been submitted.
        assert(op->get_target_node()->is_submitted());
        // Check that the target node has an event right after it.
        // Otherwise, if we use the generic batch-end event, it
        // *might* create a deadlock
        std::size_t target_node_id = op->get_target_node()->get_node_id();

        q->submit_external_wait_for(op->get_target_node());
      }
    }
  }


  std::shared_ptr<dag_node_event> evt;
  // Submit pre-event, if required
  if(pre_event) {
    pre_event->assign_event(q->insert_event());
  }

  // Submit actual operation
  op->dispatch(&dispatcher);

  // Submit post-event, if required
  if(post_event) {
    evt = q->insert_event();
    post_event->assign_event(evt);
  }

  // Mark node as submitted
  if(!evt) {
    // Use generic multi event for this batch
    evt = generic_batch_event;
  }
  node->mark_submitted(evt);

  // Remember that this was the (currently) last node submitted to the give
  // queue
  final_nodes[q] = node;
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
