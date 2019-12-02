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

#include "CL/sycl/detail/application.hpp"
#include "CL/sycl/detail/scheduling/hints.hpp"
#include "CL/sycl/detail/scheduling/device_id.hpp"
#include "CL/sycl/detail/scheduling/dag_scheduler.hpp"
#include "CL/sycl/detail/scheduling/dag_enumerator.hpp"
#include "CL/sycl/detail/scheduling/dag.hpp"
#include "CL/sycl/detail/scheduling/dag_expander.hpp"
#include "CL/sycl/detail/scheduling/operations.hpp"


namespace cl {
namespace sycl {
namespace detail {

namespace {

struct scheduling_state 
{
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
    int current_device_index,
    int current_node_index,
    Handler h) 
{
  h();

  if(current_node_index < is_device_predetermined.size()) {
    for_all_device_assignments(devices_to_try, is_device_predetermined,
                               current_device_index, current_node_index + 1, h);
  }
}

template<class Handler>
void for_all_device_assignments(
    const std::vector<device_id> &devices_to_try,
    const std::vector<char> &is_device_predetermined, 
    Handler h)
{
  for_all_device_assignments(devices_to_try, is_device_predetermined, 0, 0, h);
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
  // Start by enumerating all nodes in the dag
  dag_enumerator enumerator{d};

  // Next, create scheduling state and extract already present information
  // regarding the target device
  scheduling_state initial_state{enumerator};

  std::vector<char> is_device_predetermined(
      enumerator.get_node_index_space_size(), false);

  d->for_each_node([&](dag_node_ptr node) {
    // for each node, if it comes with information on which device to execute...
    if (node->get_execution_hints().has_hint(
            execution_hint_type::bind_to_device)) {

      assert(node->get_execution_hints().has_hint(
          execution_hint_type::dag_enumeration_id));

      // ... get the node id
      std::size_t node_id = node->get_execution_hints()
                                .get_hint<hints::dag_enumeration_id>()
                                ->id();

      // remember that the node with this id is *not* free
      // to be scheduled to arbitrary devices
      is_device_predetermined[node_id] = true;
      // instead move the supplied target device to the scheduling annotations
      initial_state.scheduling_annotations[node_id].set_target_device(
          node->get_execution_hints()
              .get_hint<hints::bind_to_device>()
              ->get_device_id());
    }
  });


  scheduling_state best_state = initial_state;
  cost_type best_cost = std::numeric_limits<cost_type>::max();

  


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
