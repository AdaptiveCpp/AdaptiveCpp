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

#include <vector>
#include <limits>
#include <unordered_map>
#include <unordered_set>

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/dag_scheduler.hpp"
#include "hipSYCL/runtime/dag_enumerator.hpp"
#include "hipSYCL/runtime/dag.hpp"
#include "hipSYCL/runtime/dag_expander.hpp"
#include "hipSYCL/runtime/dag_interpreter.hpp"
#include "hipSYCL/runtime/dag_time_table.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl {
namespace rt {

namespace {

struct scheduling_state 
{
  scheduling_state(const std::vector<node_scheduling_annotation> &annotations, 
                   const dag_enumerator &enumerator)
      : scheduling_annotations{annotations}, expansion_result{enumerator} {}

  scheduling_state(const dag_enumerator &enumerator)
      : scheduling_annotations(enumerator.get_node_index_space_size()),
        expansion_result(enumerator) {}

  std::vector<node_scheduling_annotation> scheduling_annotations;
  dag_expansion_result expansion_result;
};

template <class Handler>
void for_all_device_assignments(
    const std::vector<device_id> &devices_to_try,
    const std::vector<char> &is_device_predetermined, 
    std::vector<node_scheduling_annotation>& current_state,
    int current_device_index,
    int current_node_index,
    Handler h) 
{
  // Make sure we're not accessing invalid nodes
  if(current_node_index < is_device_predetermined.size()) {

    // If this device is pretermined, simply move to the next node
    if(is_device_predetermined[current_node_index]) {
      for_all_device_assignments(devices_to_try, is_device_predetermined,
                                 current_state, current_device_index,
                                 current_node_index + 1, h);
    } else {
      if(current_device_index < devices_to_try.size()) {
        // If we are at a valid device, put in the current scheduling
        // configuration
        current_state[current_node_index].set_target_device(
            devices_to_try[current_device_index]);

        // run the handler
        h(current_state);

        // Proceed trying out the next device on the same node
        for_all_device_assignments(devices_to_try, is_device_predetermined,
                                 current_state, current_device_index + 1, 
                                 current_node_index, h);
      } else {
        // No more devices to try on this node, move to the next node
        for_all_device_assignments(devices_to_try, is_device_predetermined,
                                 current_state, 0, 
                                 current_node_index + 1, h);
      }
    }
  }
}

template<class Handler>
void for_all_device_assignments(
    const scheduling_state& initial_state,
    const std::vector<device_id> &devices_to_try,
    const std::vector<char> &is_device_predetermined, 
    Handler h)
{
  std::vector<node_scheduling_annotation> current_state =
      initial_state.scheduling_annotations;

  // invoke handler on initial state
  h(current_state);

  for_all_device_assignments(devices_to_try, is_device_predetermined,
                             current_state, 0, 0, h);
}

void assign_execution_lanes(const dag_interpreter& d, scheduling_state& s)
{
  std::unordered_map<backend_executor*, std::size_t> _assigned_kernels;
  std::unordered_map<backend_executor*, std::size_t> _assigned_memcopies;

  d.for_each_effective_node([&](dag_node_ptr node) {

    std::size_t node_id = node->get_node_id();

    device_id target_dev = s.scheduling_annotations[node_id].get_target_device();

    backend_executor *executor =
        application::get_backend(target_dev.get_backend())
            .get_executor(target_dev);
    
    // Naive assignment algorithm for now...
    bool is_memcpy = true;
    d.for_each_operation(node, [&](operation *op) {
      // There should be no more implicit operations in the effective DAG
      // at this point. They should all have been converted into actual
      // data transfers or be removed by now
      assert(!op->is_requirement());

      if(!dynamic_is<memcpy_operation>(op)){
        is_memcpy = false;
      }
    });


    std::size_t lane = 0;
    if(is_memcpy) {
      std::size_t num_available_lanes =
          executor->get_memcpy_execution_lane_range(target_dev).num_lanes;
      lane = _assigned_memcopies[executor] % num_available_lanes +
             executor->get_memcpy_execution_lane_range(target_dev).begin;

      ++_assigned_memcopies[executor];
    }
    else {
      std::size_t num_available_lanes =
          executor->get_kernel_execution_lane_range(target_dev).num_lanes;
      lane = _assigned_kernels[executor] % num_available_lanes +
             executor->get_kernel_execution_lane_range(target_dev).begin;
      
      ++_assigned_kernels[executor];
    }

    s.scheduling_annotations[node_id].assign_to_execution_lane(executor, lane);
  });
}

void insert_synchronization_ops(const dag_interpreter& d, scheduling_state& s)
{
  d.for_each_effective_node([&](dag_node_ptr node) {
    assert(node->has_node_id());
    std::size_t node_id = node->get_node_id();

    std::vector<dag_node_ptr> synchronized_reqs;

    d.for_each_requirement(node, [&](dag_node_ptr req) {
      synchronized_reqs.push_back(req);
    });

    for(dag_node_ptr req : synchronized_reqs){
      assert(req);
    
      std::size_t req_id = req->get_node_id();

      node_scheduling_annotation &node_annotations =
          s.scheduling_annotations[node_id];
      node_scheduling_annotation &req_annotations =
          s.scheduling_annotations[req_id];

      backend_executor* executor = node_annotations.get_executor();
      backend_executor* req_executor = req_annotations.get_executor();

      if (executor == req_annotations.get_executor()) {
        if(node_annotations.get_execution_lane() ==
          req_annotations.get_execution_lane()) {

          node_annotations.add_synchronization_op(
              executor->create_wait_for_node_same_lane(node, node_annotations,
                                                      req, req_annotations));
        }
        else {
          node_annotations.add_synchronization_op(
              executor->create_wait_for_node_same_backend(
                  node, node_annotations, req, req_annotations));
        }
      } else if (node_annotations.get_target_device().get_backend() ==
                req_annotations.get_target_device().get_backend()) {
        node_annotations.add_synchronization_op(
            executor->create_wait_for_node_same_backend(node, node_annotations,
                                                        req, req_annotations));
      } else {
        // Snychronization between different backends always requires inserting
        // an event after the node that we wait for.
        req_annotations.insert_event_after_if_missing(req_executor->create_event_after(req));
        node_annotations.add_synchronization_op(
            executor->create_wait_for_external_node(node, node_annotations, req,
                                                    req_annotations));
      }
    
    }
  });
}

// The dag_expander needs to know for each memory requirement where it will
// be accessed in order to correctly calculate conflicts or node merge
// opportunities.
// This means that we have to assign memory requirements to the same device
// as the kernel/operation requiring them.
void assign_requirements_to_devices(scheduling_state &state, const dag *d)
{
  d->for_each_node([&](dag_node_ptr node) {
    if (!node->get_operation()->is_requirement()) {

      std::size_t node_id = node->get_node_id();

      for (auto req : node->get_requirements()) {
        // Do not assign devices if other requirement is a direct/explicit
        // dependency on some other operation
        if (req->get_operation()->is_requirement()) {
          if (!req->is_submitted()) {
            std::size_t req_id = req->get_node_id();

            state.scheduling_annotations[req_id].set_target_device(
                state.scheduling_annotations[node_id].get_target_device());
          }
        }
      }
    }
  });
}

cost_type evaluate_scheduling_decisions(const scheduling_state &state,
                                        const dag_interpreter& interpreter,
                                        const dag_enumerator &enumerator)
{
  dag_time_table time_table{interpreter, enumerator, state.scheduling_annotations};
  return time_table.get_expected_total_dag_duration();
}

void assign_memcopies_to_devices(const dag_interpreter &interpreter,
                                 scheduling_state &state)
{
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    if (node->get_operation()->is_data_transfer()) {
      std::size_t node_id = node->get_node_id();

      assert(dynamic_is<memcpy_operation>(node->get_operation()));

      memcpy_operation *op = cast<memcpy_operation>(node->get_operation());

      device_id assigned_device;
      // If a non-CPU backend is involved, use this backend.
      // The logic behind this is that GPU backends are generally
      // capable to copy between host and GPU, while pure CPU backends
      // generally cannot transfer data between host and device.
      // So, as soon as one non-CPU backend is involved, it must
      // be used for memory copies.

      // The case where two non-CPU backends should interact is not yet
      // supported
      hardware_platform source_platform =
          op->source().get_device().get_full_backend_descriptor().hw_platform;
      hardware_platform dest_platform =
          op->dest().get_device().get_full_backend_descriptor().hw_platform;
      
      if (source_platform != hardware_platform::cpu &&
          dest_platform != hardware_platform::cpu &&
          source_platform != dest_platform)
        assert(false &&
               "Interactions between two non-CPU backends are unimplemented");

      if (source_platform != hardware_platform::cpu)
        assigned_device = op->source().get_device();
      else
        assigned_device = op->dest().get_device();

      state.scheduling_annotations[node_id].set_target_device(assigned_device);
    }
  });  
}

} // anonymous namespace

dag_scheduler::dag_scheduler()
{
  // Collect available devices (currently just uses everything)
  // TODO: User may want to restrict the device to which we schedule

  application::get_runtime().backends().for_each_backend(
  [this](backend *b) {
    std::size_t num_devices = b->get_hardware_manager()->get_num_devices();
    for(std::size_t dev = 0; dev < num_devices; ++dev){

      this->_available_devices.push_back(
          b->get_hardware_manager()->get_device_id(dev));

    }
  });
}

void dag_scheduler::submit(dag* d)
{
  // This should also be checked at a higher level,
  // throwing an exception such that the user can handle
  // this error
  assert(_available_devices.size() > 0 && "No devices available");

  // Start by enumerating all nodes in the dag
  dag_enumerator enumerator{d};

  // Next, create scheduling state and extract already present information
  // regarding the target device
  auto initial_state = std::make_unique<scheduling_state>(enumerator);

  std::vector<char> is_device_predetermined(
      enumerator.get_node_index_space_size(), false);

  d->for_each_node([&](dag_node_ptr node) {
    assert(node->has_node_id());

    // ... get the node id
    std::size_t node_id = node->get_node_id();

    // for each node, if it comes with information on which device to execute...
    if (node->get_execution_hints().has_hint(
            execution_hint_type::bind_to_device)) {

      // remember that the node with this id is *not* free
      // to be scheduled to arbitrary devices
      is_device_predetermined[node_id] = true;
      // instead move the supplied target device to the scheduling annotations
      initial_state->scheduling_annotations[node_id].set_target_device(
          node->get_execution_hints()
              .get_hint<hints::bind_to_device>()
              ->get_device_id());
    } else {
      // Requirements are not free to be executed on arbitrary devices,
      // (they are bound to the same device as the kernel to which they belong)
      // so we mark them as predetermined
      // Memory copies are also not free to be scheduled to arbitrary devices.
      if (node->get_operation()->is_requirement() ||
          node->get_operation()->is_data_transfer())
        is_device_predetermined[node_id] = true;

      // Start with the first device
      initial_state->scheduling_annotations[node_id].set_target_device(
          _available_devices[0]);
    }
  });

  auto best_state = std::make_unique<scheduling_state>(
      initial_state->scheduling_annotations, enumerator);

  cost_type best_cost = std::numeric_limits<cost_type>::max();

  dag_expander expander{d, enumerator};

  for_all_device_assignments(
      *initial_state, _available_devices, is_device_predetermined,
      [&](const std::vector<node_scheduling_annotation> &current_state)
  {
    auto candidate_state =
        std::make_unique<scheduling_state>(current_state, enumerator);

    assign_requirements_to_devices(*candidate_state, d);

    expander.expand(candidate_state->scheduling_annotations,
                    candidate_state->expansion_result);

    dag_interpreter interpreter{d, &enumerator,
                                &candidate_state->expansion_result};

    // We need to correct actual required memory copies and bind them
    // to the backend that can actual execute them (e.g. CPU backends
    // usually cannot migrate data to the host from GPUs)
    assign_memcopies_to_devices(interpreter, *candidate_state);

    assign_execution_lanes(interpreter, *candidate_state);

    // TODO We should also assign some proposal for the execution order
    insert_synchronization_ops(interpreter, *candidate_state);
    // TODO This does not yet take into account queuing on a single execution lane!
    cost_type c = evaluate_scheduling_decisions(*candidate_state, interpreter,
                                                enumerator);

    if(c < best_cost) {
      best_cost = c;
      best_state = std::move(candidate_state);
    }
  });

  // Apply new memory states
  for (std::size_t i = 0; 
      i < enumerator.get_data_region_index_space_size();
      ++i) {

    best_state->expansion_result.original_data_region(i)->apply_fork(
      best_state->expansion_result.memory_state(i));
  }

  // Register users of buffers
  // TODO: Could we also invoke that function earlier, e.g. in the dag_builder,
  // since the nodes themselves and their dependencies shouldn't change during
  // scheduling?
  d->commit_node_dependencies();

  // Assign final data to nodes
  // TODO: Maybe we could reuse the dag_interpreter of the best state?
  dag_interpreter interpreter{d, &enumerator, &best_state->expansion_result};
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    std::size_t node_id = node->get_node_id();

    const node_scheduling_annotation &annotation =
        best_state->scheduling_annotations[node_id];

    node->assign_to_device(annotation.get_target_device());
    node->assign_to_executor(annotation.get_executor());
    node->assign_to_execution_lane(annotation.get_execution_lane());
  });
  // Initialize deferred pointers, i.e. bind accessors to actual data pointers
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    for (dag_node_ptr req : node->get_requirements()) {
      if (req->get_operation()->is_requirement()) {
        if (cast<requirement>(req->get_operation())->is_memory_requirement()) {
          memory_requirement *mem_req =
              cast<memory_requirement>(req->get_operation());
          if (mem_req->is_buffer_requirement()) {
            buffer_memory_requirement *bmem_req =
                cast<buffer_memory_requirement>(mem_req);

            device_id target_dev = node->get_assigned_device();
            void* device_pointer = bmem_req->get_data_region()->get_memory(target_dev);
            bmem_req->initialize_device_data(device_pointer);
          }
        }
      }
    }
  });

  // Emit nodes to backend executors

  // Find all unique backend executors
  std::unordered_set<backend_executor *> unique_executors;
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    unique_executors.insert(node->get_assigned_executor());
  });

  for (backend_executor *executor : unique_executors) {
    executor->submit_dag(interpreter, enumerator,
                         best_state->scheduling_annotations);
  }

  // If all went well, the backend executor should have marked all nodes as
  // submitted
  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    assert(node->is_submitted());
  });

  // Create virtual events for requirements in case user wants to wait
  // explicitly for requirements
  d->for_each_node([&](dag_node_ptr node){
    if(!node->get_event()) {
      // All nodes that have events should have been submitted
      assert(!node->is_submitted());

      // Create virtual events if there are no events yet
      std::vector<std::shared_ptr<dag_node_event>> effective_requirements;
      interpreter.for_each_requirement(node, [&](dag_node_ptr req){
        effective_requirements.push_back(req->get_event());
      });

      node->mark_submitted(std::make_shared<dag_multi_node_event>(effective_requirements));

      assert(node->get_event() != nullptr);
    }
  });

  // Register nodes as submitted
  application::dag().register_submitted_ops(interpreter);
}

}
}
