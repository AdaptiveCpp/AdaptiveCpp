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

#include <limits>
#include <unordered_map>

#include "CL/sycl/detail/application.hpp"
#include "CL/sycl/detail/scheduling/util.hpp"
#include "CL/sycl/detail/scheduling/hints.hpp"
#include "CL/sycl/detail/scheduling/device_id.hpp"
#include "CL/sycl/detail/scheduling/dag_scheduler.hpp"
#include "CL/sycl/detail/scheduling/dag_enumerator.hpp"
#include "CL/sycl/detail/scheduling/dag.hpp"
#include "CL/sycl/detail/scheduling/dag_expander.hpp"
#include "CL/sycl/detail/scheduling/dag_interpreter.hpp"
#include "CL/sycl/detail/scheduling/operations.hpp"


namespace cl {
namespace sycl {
namespace detail {

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
    assert(node->get_execution_hints().has_hint(
        execution_hint_type::dag_enumeration_id));

    std::size_t node_id =
        node->get_execution_hints().get_hint<hints::dag_enumeration_id>()->id();

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
          executor->get_memcpy_execution_lane_range().num_lanes;
      lane = _assigned_memcopies[executor] % num_available_lanes +
             executor->get_memcpy_execution_lane_range().begin;

      ++_assigned_memcopies[executor];
    }
    else {
      std::size_t num_available_lanes =
          executor->get_kernel_execution_lane_range().num_lanes;
      lane = _assigned_kernels[executor] % num_available_lanes +
             executor->get_kernel_execution_lane_range().begin;
      
      ++_assigned_kernels[executor];
    }

    s.scheduling_annotations[node_id].assign_to_execution_lane(executor, lane);
  });
}

void insert_synchronization_ops(const dag_interpreter& d, scheduling_state& s)
{
  d.for_each_effective_node([&](dag_node_ptr node) {
    assert(node->get_execution_hints().has_hint(
        execution_hint_type::dag_enumeration_id));
    std::size_t node_id =
        node->get_execution_hints().get_hint<hints::dag_enumeration_id>()->id();

    d.for_each_requirement(node, [&](dag_node_ptr req) {

      std::size_t req_id = req->get_execution_hints()
                               .get_hint<hints::dag_enumeration_id>()
                               ->id();

      node_scheduling_annotation &node_annotations =
          s.scheduling_annotations[node_id];
      const node_scheduling_annotation &req_annotations =
          s.scheduling_annotations[req_id];

      if ((node_annotations.get_executor() == req_annotations.get_executor()) &&
          (node_annotations.get_execution_lane() ==
           req_annotations.get_execution_lane())) {

        //node_annotations.add_synchronization_op(std::unique_ptr<backend_synchronization_operation> op)
      }
    });
  });
}

cost_type evaluate_scheduling_decisions(const scheduling_state &state,
                                        const dag *d,
                                        const dag_enumerator &enumerator)
{
  

  // TODO
  return 0.0;
}

} // anonymous namespace

dag_scheduler::dag_scheduler()
{
  // Collect available devices (currently just uses everything)
  // TODO: User may want to restrict the device to which we schedule

  application::get_hipsycl_runtime().backends().for_each_backend(
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
    assert(node->get_execution_hints().has_hint(
          execution_hint_type::dag_enumeration_id));

    // ... get the node id
    std::size_t node_id = node->get_execution_hints()
                              .get_hint<hints::dag_enumeration_id>()
                              ->id();

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
      if(node->get_operation()->is_requirement())
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
    [&](const std::vector<node_scheduling_annotation> &current_state) {

    // TODO Fix requirement device assignments?

    auto candidate_state = std::make_unique<scheduling_state>(
        current_state, enumerator);
    
    expander.expand(candidate_state->scheduling_annotations,
                    candidate_state->expansion_result);

    dag_interpreter interpreter{d, &enumerator,
                                &candidate_state->expansion_result};

    assign_execution_lanes(interpreter, *candidate_state);
    insert_synchronization_ops(interpreter, *candidate_state);

    cost_type c = evaluate_scheduling_decisions(*candidate_state, d, enumerator);

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

  // Emit nodes to backend executor
}

} // detail
} // sycl
} // cl
