/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#include <cassert>

#include "hipSYCL/runtime/inorder_executor.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"

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

  virtual result dispatch_kernel(kernel_operation *op,
                                 dag_node_ptr node) final override {

    return _queue->submit_kernel(*op, node);
  }

  virtual result dispatch_memcpy(memcpy_operation *op,
                                 dag_node_ptr node) final override {
    return _queue->submit_memcpy(*op, node);
  }

  virtual result dispatch_prefetch(prefetch_operation *op,
                                   dag_node_ptr node) final override {
    return _queue->submit_prefetch(*op, node);
  }

  virtual result dispatch_memset(memset_operation *op,
                                 dag_node_ptr node) final override {
    return _queue->submit_memset(*op, node);
  }

private:
  inorder_queue* _queue;
};

std::size_t
get_maximum_execution_index_for_lane(const std::vector<dag_node_ptr> &nodes,
                                     inorder_queue* lane) {
  std::size_t index = 0;
  for (const auto &node : nodes) {
    if (node->is_submitted() &&
        node->get_assigned_device().get_backend() ==
            lane->get_device().get_backend() &&
        node->get_assigned_execution_lane() == lane) {
      if(node->get_assigned_execution_index() > index)
        index = node->get_assigned_execution_index();
    }
  }
  return index;
}

} // anonymous namespace

inorder_executor::inorder_executor(std::unique_ptr<inorder_queue> q)
: _q{std::move(q)}, _num_submitted_operations{0} {}

inorder_executor::~inorder_executor(){}

bool inorder_executor::is_inorder_queue() const {
  return true;
}

bool inorder_executor::is_outoforder_queue() const {
  return false;
}

bool inorder_executor::is_taskgraph() const {
  return false;
}

void inorder_executor::submit_directly(dag_node_ptr node, operation *op,
                                       const std::vector<dag_node_ptr> &reqs) {
  
  HIPSYCL_DEBUG_INFO << "inorder_executor: Processing node " << node.get()
	  << " with " << reqs.size() << " non-virtual requirement(s) and "
	  << node->get_requirements().size() << " direct requirement(s)." << std::endl;

  assert(!op->is_requirement());

  if (node->is_submitted())
    return;

  node->assign_to_execution_lane(_q.get());

  node->assign_execution_index(_num_submitted_operations);
  ++_num_submitted_operations;

  // Submit synchronization mechanisms
  result res;
  for (auto req : reqs) {
    // The scheduler should not hand us virtual requirements
    assert(!req->is_virtual());
    assert(req->is_submitted());

    // Nothing to do if we have to synchronize with
    // an operation that is already known to have completed
    if(!req->is_known_complete()) {
      if (req->get_assigned_device().get_backend() !=
          _q->get_device().get_backend()) {
        HIPSYCL_DEBUG_INFO
            << " --> Synchronizes with external node: " << req
            << std::endl;
        res = _q->submit_external_wait_for(req);
      } else {
        if (req->get_assigned_execution_lane() == _q.get()) {
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
                            << std::endl;
          // We only need to actually synchronize with the lane if this req
          // is the operation that has been submitted *last* to the lane
          // out of all requirements in reqs.
          // (Follows from execution lanes being in-order queues)
          //
          // Find the maximum execution index out of all our requirements.
          // Since the execution index is incremented after each submission,
          // this allows us to identify the requirement that was submitted last.
          inorder_queue *req_q = static_cast<inorder_queue *>(
              req->get_assigned_execution_lane());
          std::size_t maximum_execution_index =
              get_maximum_execution_index_for_lane(reqs, req_q);
          
          if(req->get_assigned_execution_index() != maximum_execution_index) {
            HIPSYCL_DEBUG_INFO
                << "  --> (Skipping unnecessary synchronization; another "
                   "requirement follows in the same inorder queue)"
                << std::endl;
          } else {
            res = _q->submit_queue_wait_for(req);
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
      << "inorder_executor: Dispatching to lane " << _q.get() << ": "
      << dump(op) << std::endl;
  
  queue_operation_dispatcher dispatcher{_q.get()};
  res = op->dispatch(&dispatcher, node);
  if (!res.is_success()) {
    register_error(res);
    node->cancel();
    return;
  }

  if (node->get_execution_hints()
          .has_hint<hints::coarse_grained_synchronization>()) {
    node->mark_submitted(_q->create_queue_completion_event());
  } else {
    node->mark_submitted(_q->insert_event());
  }
}

inorder_queue* inorder_executor::get_queue() const {
  return _q.get();
}

bool inorder_executor::can_execute_on_device(const device_id& dev) const {
  return _q->get_device() == dev;
}

bool inorder_executor::is_submitted_by_me(dag_node_ptr node) const {
  if(!node->is_submitted())
    return false;
  return node->get_assigned_executor() == this;
}

}
}