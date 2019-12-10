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

#include "CL/sycl/detail/scheduling/dag_time_table.hpp"


namespace cl {
namespace sycl {
namespace detail {

namespace {

cost_type get_runtime_cost(const dag_interpreter &interpreter,
                           const node_scheduling_annotation& annotation,
                           dag_node_ptr node) 
{
  cost_type c = 0.0;
  interpreter.for_each_operation(node, [&](operation* op) {
    // Assume for now that all operations of a node are serialized
    // (the case of a dag node consisting of multiple ops is rare anyway)
    c += op->get_runtime_costs();
  });

  std::size_t node_id =
      node->get_execution_hints().get_hint<hints::dag_enumeration_id>()->id();
  
  cost_type max_synchronization_duration = 0.0;
  for(auto synchronization_op : annotation.get_synchronization_ops()) {
    // We assume for now that the execution backend can overlap simultaneous
    // synchronization/wait operations (this may not be the case on HIP/CUDA!)
    // Since wait() operations take place before the node execution,
    // the elapsed time will be the maximum of the duration of all synchronization
    // ops before the node itself is processed.
    // The same is true for operations after node execution, but since only
    // \c event_after_node is runs after node execution, we can just add this duration
    // in any case
    //
    // TODO: Query the backend for overlapping synchronization capabilities

    if(synchronization_op->is_event_after_node())
      c += synchronization_op->get_runtime_costs();
    else {
      cost_type duration = synchronization_op->get_runtime_costs();
      if(duration > max_synchronization_duration)
        max_synchronization_duration = duration;
    }
  }

  c += max_synchronization_duration;

  return c;
}

double
get_node_start_date(const dag_interpreter &interpreter,
                    const std::vector<node_scheduling_annotation> &annotations,
                    dag_node_ptr node)
{

}

}

dag_time_table::dag_time_table(
      const dag_interpreter& interpreter,
      const dag_enumerator& enumerator,
      const std::vector<node_scheduling_annotation> &scheduling_annotations)
  : _time_ranges(enumerator.get_node_index_space_size())
{
  
}

}
}
}