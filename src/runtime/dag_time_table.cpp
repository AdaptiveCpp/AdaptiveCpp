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

#include "hipSYCL/runtime/dag_time_table.hpp"
#include <algorithm>


namespace hipsycl {
namespace rt {

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
get_node_duration(const dag_interpreter &interpreter,
                  const std::vector<node_scheduling_annotation> &annotations,
                  std::vector<dag_time_table::time_range> &time_ranges,
                  dag_node_ptr node)
{
  if (node->is_submitted())
    return 0.0;

  std::size_t node_id = node->get_node_id();

  if (time_ranges[node_id].duration == -1.0) {
    time_ranges[node_id].duration =
        get_runtime_cost(interpreter, annotations[node_id], node);
  }

  return time_ranges[node_id].duration;
}


std::size_t get_assigned_execution_lane(
    std::size_t node_id,
    const std::vector<node_scheduling_annotation> &annotations)
{
  return annotations[node_id].get_execution_lane();
}

std::size_t get_assigned_execution_lane(
    dag_node_ptr node,
    const std::vector<node_scheduling_annotation> &annotations)
{

  if (node->is_submitted()) {
    return 0;
  }
  
  std::size_t node_id = node->get_node_id();
  
  return get_assigned_execution_lane(node_id, annotations);
}


double
get_node_start_date(const dag_interpreter &interpreter,
                    const std::vector<node_scheduling_annotation> &annotations,
                    std::vector<dag_time_table::time_range> &time_ranges,
                    dag_node_ptr node)
{
  
  std::size_t node_id = node->get_node_id();
  
  if (node->is_submitted()) {
    // TODO Would it make sense to record actual start dates for nodes
    // when submitting them?
    return 0.0;
  } else {
    if (time_ranges[node_id].start == -1.0) {
      double max_end_date = 0.0;
      interpreter.for_each_requirement(node, [&](dag_node_ptr req) {
        double requirement_start_date =
            get_node_start_date(interpreter, annotations, time_ranges, req);

        double requirement_end_date =
            requirement_start_date +
            get_node_duration(interpreter, annotations, time_ranges, node);

        if (requirement_end_date > max_end_date)
          max_end_date = requirement_end_date;
      });
      // In addition to all explicit requirements, we also need to consider all
      // nodes that are before this node on the same execution lane

      /* TODO - This requires the introduction of the node order scheduling annotation */
      /* TODO - Need to introduce execution lanes */

      time_ranges[node_id].start = max_end_date;
      time_ranges[node_id].duration =
          get_node_duration(interpreter, annotations, time_ranges, node);

    }
  }
  return time_ranges[node_id].start;
}

} // anonymous namespace

dag_time_table::dag_time_table(
      const dag_interpreter& interpreter,
      const dag_enumerator& enumerator,
      const std::vector<node_scheduling_annotation> &scheduling_annotations)
  : _time_ranges(enumerator.get_node_index_space_size())
{
  for (auto &t : _time_ranges) {
    // get_node_start_date() recognize a node as unprocessed if
    // the start time is -1
    t.start = -1;
    t.duration = -1;
  }

  interpreter.for_each_effective_node([&](dag_node_ptr node) {
    std::size_t node_id = node->get_node_id();

    this->_time_ranges[node_id].start = get_node_start_date(
        interpreter, scheduling_annotations, this->_time_ranges, node);
    this->_time_ranges[node_id].duration = get_node_duration(
        interpreter, scheduling_annotations, this->_time_ranges, node);
  });
}

double dag_time_table::get_expected_total_dag_duration() const
{

  if (_time_ranges.empty())
    return 0.0;
  
  auto it =
      std::max_element(_time_ranges.begin(), _time_ranges.end(),
                       [](const time_range &a, const time_range &b) {
                         return a.start + a.duration < b.start + b.duration;
                       });

  assert(it != _time_ranges.end());
  return it->start + it->duration;
}

}
}
