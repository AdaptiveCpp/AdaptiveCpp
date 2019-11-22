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

#include <unordered_set>
#include <algorithm>

#include "CL/sycl/detail/scheduling/dag_expander.hpp"
#include "CL/sycl/detail/scheduling/util.hpp"
#include "CL/sycl/detail/debug.hpp"

namespace cl {
namespace sycl {
namespace detail {

namespace {

bool is_memory_requirement(const dag_node_ptr& node)
{
  if(!node->get_operation()->is_requirement())
    return false;

  if(!cast<requirement>(node->get_operation())->is_memory_requirement())
    return false;

  return true;
}

bool access_ranges_overlap(sycl::id<3> offset_a, sycl::range<3> range_a,
                          sycl::id<3> offset_b, sycl::range<3> range_b)
{
  sycl::id<3> a_min = offset_a;
  sycl::id<3> a_max = offset_a + sycl::id<3>{range_a[0],range_a[1],range_a[2]};

  sycl::id<3> b_min = offset_b;
  sycl::id<3> b_max = offset_b + sycl::id<3>{range_b[0],range_b[1],range_b[2]};

  for(int dim = 0; dim < 3; ++dim){
    if(std::max(a_min[dim], b_min[dim]) > std::min(a_max[dim], b_max[dim]))
      return false;
  }
  return true;
}

// Tests if the access range a is >= the access range described by b.
bool access_range_is_greater_or_equal(sycl::id<3> offset_a, sycl::range<3> range_a,
                                    sycl::id<3> offset_b, sycl::range<3> range_b)

{
  sycl::id<3> a_min = offset_a;
  sycl::id<3> b_min = offset_b;
  sycl::id<3> a_max = a_min + sycl::id<3>{range_a[0],range_a[1],range_a[2]};
  sycl::id<3> b_max = b_min + sycl::id<3>{range_b[0],range_b[1],range_b[2]};

  for(int dim = 0; dim < 3; ++dim) {
    if(a_min[dim] > b_min[dim])
      return false;

    if(a_max[dim] < b_max[dim])
      return false;
  }

  return true;
}

bool is_overlapping_memory_requirement(
  memory_requirement* a,
  memory_requirement* b)
{
  if(a->is_buffer_requirement() && b->is_buffer_requirement()){
    buffer_memory_requirement* buff_a = cast<buffer_memory_requirement>(a);
    buffer_memory_requirement* buff_b = cast<buffer_memory_requirement>(b);

    if(buff_a->get_data_region() != buff_b->get_data_region())
      return false;
    
    if(!access_ranges_overlap(buff_a->get_access_offset3d(),
                              buff_a->get_access_range3d(),
                              buff_b->get_access_offset3d(),
                              buff_b->get_access_range3d()))
      return false;

    return true;
  }
  else {
    assert(false && "Non-buffer requirements are unimplemented");
  }

  return false;
}

device_id get_assigned_device(dag_node_ptr node)
{
  // We require that the scheduler has already bound 
  // all nodes to devices
  assert(node->get_execution_hints().has_hint(
    execution_hint_type::bind_to_device));

  return node->get_execution_hints()
    .get_hint<hints::bind_to_device>()->get_device_id();
}

/// Check if \c requirement has an equivalent memory access
/// (e.g., memory requirement) compared to \c other, such that
/// the memory access in \c node can be optimized away.
bool is_requirement_satisfied_by(
  dag_node_ptr requirement,
  dag_node_ptr other)
{
  if(is_memory_requirement(requirement) && is_memory_requirement(other)){
    if(get_assigned_device(requirement) == get_assigned_device(other)) {

      if(cast<memory_requirement>(requirement->get_operation())
          ->is_buffer_requirement() &&
         cast<memory_requirement>(other->get_operation())
          ->is_buffer_requirement()){

        auto buff_req = cast<buffer_memory_requirement>(requirement->get_operation());
        auto other_buff_req = cast<buffer_memory_requirement>(other->get_operation());

        auto region = buff_req->get_data_region();
        auto other_region = other_buff_req->get_data_region();

        // Check if both requirements are referring to the same underlying
        // buffer
        if(region == other_region) {
          // If the access range of the other requirement covers the access range
          // of \c requirement (or is larger), then \c requirement can also
          // be satisfied by the other requirement
          if(access_range_is_greater_or_equal(other_buff_req->get_access_offset3d(),
                                              other_buff_req->get_access_range3d(),
                                              buff_req->get_access_offset3d(),
                                              buff_req->get_access_range3d()))
            return true;

        }

      }
    }
  }
  else {
    // TODO Check for superfluous explicit copy operations?
    // TODO Explicit copy operations may also satisfy a requirement
  }
  return false;
}

void order_by_requirements_recursively(
  dag_node_ptr current,
  std::vector<dag_node_ptr>& nodes_to_process,
  std::vector<dag_node_ptr>& out)
{
  // A node that has already been submitted belongs to a DAG
  // that has already been fully constructed (and executed), 
  // and hence cannot be in the expansion process that the
  // dag_expander implements -- we skip it in order to avoid
  // descending into the far past of nodes that have been
  // executed a long time ago
  if(current->is_submitted())
    return;

  auto node_iterator = 
  std::find(nodes_to_process.begin(), 
            nodes_to_process.end(), current);

  // Abort if we are a memory requirement that is
  // not contained in the list of nodes
  // that still need processing
  if(node_iterator == nodes_to_process.end())
    return;
  else
  {
    // Remove node from the nodes that need processing
    nodes_to_process.erase(node_iterator);

    // First, process any requirements to make sure
    // they are listed in the output before this node
    for(auto req : current->get_requirements())
      order_by_requirements_recursively(req, nodes_to_process, out);

    out.push_back(current);
  }

}



/// In a given range in a ordered memory requirements list, 
/// finds the maximum possible range where a candidate mergeable
/// with \c begin might occur.
/// Inside the range given by \c begin and the return value of this function,
/// it is guaranteed that requirement nodes pointing to the the same memory
/// can be safely merged.
std::vector<dag_node_ptr>::const_iterator
find_max_merge_candidate_range(std::vector<dag_node_ptr>::const_iterator begin,
                              std::vector<dag_node_ptr>::const_iterator end)
{
  assert(is_memory_requirement(*begin));
  
  device_id begin_device = get_assigned_device(*begin);

  buffer_memory_requirement* begin_mem_req = 
    cast<buffer_memory_requirement>((*begin)->get_operation());
  
  for(auto it = begin; it != end; ++it){
    assert(is_memory_requirement(*it));

    memory_requirement* mem_req = 
      cast<memory_requirement>((*it)->get_operation());

    // There cannot be any nodes mergable with begin
    // after it, if
    // * it refers to the same memory as begin
    // * it is accessed on a different device as begin
    if(mem_req->is_buffer_requirement()){
      auto buff_mem_req = cast<buffer_memory_requirement>(mem_req);

      if(is_overlapping_memory_requirement(buff_mem_req, begin_mem_req)){

        device_id current_device = get_assigned_device(*it);

        if(current_device != begin_device)
          return it;
      }
    }
    else {
      // Image requirement is still unimplemented
    }
  }

  return end;
}

hints::dag_expander_annotation* get_or_create_hint(dag_node_ptr node)
{
  execution_hints& hints = node->get_execution_hints();
  execution_hint* h = hints.get_hint(execution_hint_type::dag_expander_annotation);

  if(h){
    return cast<hints::dag_expander_annotation>(h);
  }
  else{
    execution_hint_ptr new_hint = make_execution_hint<hints::dag_expander_annotation>();
    hints.add_hint(new_hint);

    return cast<hints::dag_expander_annotation>(new_hint.get());
  }
}

/// Identifies requirements in \c ordered_nodes that can be merged and adds
/// marks them with a "forwarded-to" dag_expander execution hint to indicate
/// that the scheduler should instead only take the node into account
/// that this node is forwarded to.
/// \param ordered_nodes A vector of nodes where the elements
/// are ordered such that all requirements precede the node in the vector for
/// all nodes in the vector.
void mark_mergeable_nodes(const std::vector<dag_node_ptr>& ordered_nodes)
{
  // Because of the order of nodes in ordered_nodes,
  // we can find opportunities for merging by looking at a given node
  // and all the succeeding nodes.
  std::unordered_set<dag_node_ptr> processed_nodes;

  for(auto mem_req_it = ordered_nodes.begin();
      mem_req_it != ordered_nodes.end();
      ++mem_req_it){

    if(is_memory_requirement(*mem_req_it) && 
      processed_nodes.find(*mem_req_it) == processed_nodes.end()){

      auto merge_candidates_end = 
        find_max_merge_candidate_range(mem_req_it, 
                                      ordered_nodes.end());

      for(auto merge_candidate = mem_req_it;
          merge_candidate != merge_candidates_end; 
          ++merge_candidate){

        if(is_memory_requirement(*merge_candidate) &&
          processed_nodes.find(*merge_candidate) == processed_nodes.end()) {

          // If we can merge the node with the candidate,
          // mark the candidate as "forwarding" to the node
          if(is_overlapping_memory_requirement(
            cast<memory_requirement>((*mem_req_it)->get_operation()),
            cast<memory_requirement>((*merge_candidate)->get_operation())))
          {
            HIPSYCL_DEBUG_INFO << "dag_expander: Marking requirement "
                << mem_req_it->get() << " for merging with node "
                << merge_candidate->get() << std::endl;

            hints::dag_expander_annotation* expander_hint = 
              get_or_create_hint(*merge_candidate);

            expander_hint->set_forward_to_node(*mem_req_it);
          }

        }
      }

      processed_nodes.insert(*mem_req_it);
    
    }
  }
}



/// Checks if a requirement node is superfluous and can be removed.
/// This is the case, if
/// * A requirement or explicit copy operation precedes 
///   that alredy accesses the same (or larger)
///   data region on the same device;
/// * No requirement of the same memory precedes, but the
///   data is already up-to-date on the target device
///
/// \param ordered_nodes The nodes in the DAG ordered such that
/// the requirements of all nodes precede the nodes in terms of
/// their index in the std::vector.
/// \param node_index The index of the requirement node.
bool can_requirement_be_removed(const std::vector<dag_node_ptr>& ordered_nodes, 
                                int node_index)
{
  assert_is<memory_requirement>(ordered_nodes[node_index]->get_operation());

  // If a node already has a dag_expander hint, it must be
  // because it has been decided in the previous phase
  // that this node should be merged into another one.
  // In this case, we cannot remove/replace it, so we skip it.
  if(!ordered_nodes[node_index]->get_execution_hints().has_hint(
    execution_hint_type::dag_expander_annotation)){

    // Obtain the device to which this requirement has
    // been bound by the scheduler
    device_id assigned_device = 
      get_assigned_device(ordered_nodes[node_index]);

    // Check if a requirement precedes this requirement
    // that makes it superfluous
    for(int j=node_index-1; j >= 0; --j) {
      // Also check here for dag_expander_annotation hint in
      // ordered_nodes[j] - if the hint is present it means
      // the node has already been processed.
      if(!ordered_nodes[j]->get_execution_hints().has_hint(
        execution_hint_type::dag_expander_annotation)) {

        if(is_memory_requirement(ordered_nodes[j])){
          // If the data is used on a different device, with
          // no requirements on the same device before,
          // this memory requirement certainly cannot be removed.
          if(assigned_device != get_assigned_device(ordered_nodes[j])) {
            if(is_overlapping_memory_requirement(
              cast<memory_requirement>(ordered_nodes[node_index]->get_operation()),
              cast<memory_requirement>(ordered_nodes[j]->get_operation()))) {
              return false;
            }
          }

          if(is_requirement_satisfied_by(ordered_nodes[node_index], 
                                        ordered_nodes[j]))
            return true;
        }

      }
    }
    // There's no equivalent requirement preceding - check
    // if data is already up-to-date on device
    if(cast<memory_requirement>(ordered_nodes[node_index]->get_operation())
      ->is_buffer_requirement()){

      auto data_region = 
        cast<buffer_memory_requirement>(ordered_nodes[node_index]->get_operation())
        ->get_data_region();

      // TODO: We do not yet have an efficient means of querying
      // whether data regions are current on a device,
      // so just force the copy for now...

      HIPSYCL_DEBUG_WARNING << "dag_expander: Scheduling potentially "
        "superfluous memory transfer due to unimplemented query for data version "
        "in n-dimensional ranges in data_range" << std::endl;
      return false;
      
      // return true;
    }
  }
  
  return false;
}

} // anonymous namespace

dag_expander::dag_expander(dag* d)
: _dag{d}
{
  // Find parents in the requirement graph
  d->for_each_node([this](dag_node_ptr node) {
    for(auto req : node->get_requirements()) {
      _parents[req].push_back(node);
    }
  });

  // 1.) As a first step, we identify requirements that
  // can be merged in a single requirement.

  // Linearize nodes to determine opportunities
  // for merging nodes

  std::vector<dag_node_ptr> ordered_nodes;

  // This fills the ordered_nodes vector with
  // all nodes from the DAG in an order
  // that guarantees that an entry in the vector only depends
  // only on entries that precede it in the vector 
  this->order_by_requirements(ordered_nodes);
  mark_mergeable_nodes(ordered_nodes);

  // 2.) Determine if non-merged nodes should be replaced
  // by an actual memory transfer or be replaced by a no-op

  for(int i = 0; i < static_cast<int>(ordered_nodes.size()); ++i) {
    // A requirement node can be _removed_, if
    // * A requirement or explicit copy operation precedes 
    //   that alredy accesses the same (or larger)
    //   data region on the same device;
    // * No requirement of the same memory precedes, but the
    //   data is already up-to-date on the target device
    // Otherwise, the requirement node must be replaced by an
    // actual memory transfer.
    
    if(is_memory_requirement(ordered_nodes[i])){
      // If a node already has a dag_expander hint, it must be
      // because it has been decided in the previous phase
      // that this node should be merged into another one.
      // In this case, we cannot remove/replace it, so we skip it.
      if(!ordered_nodes[i]->get_execution_hints().has_hint(
        execution_hint_type::dag_expander_annotation)){

        hints::dag_expander_annotation* expander_hint = 
              get_or_create_hint(ordered_nodes[i]);

        if(can_requirement_be_removed(ordered_nodes, i)) {
          HIPSYCL_DEBUG_INFO << "dag_expander: Marking requirement "
                << ordered_nodes[i].get() << " for removal"
                << std::endl;

          expander_hint->set_optimized_away();
        }
        else {
          HIPSYCL_DEBUG_INFO << "dag_expander: Replacing requirement with "
            "data transfer" << std::endl;
            
          expander_hint->set_replacement_operation();
        }
      }
    }
    else {
      // TOOD: Should we also attempt to optimize away
      // explicit copy operations?
    }
  }
}


void dag_expander::order_by_requirements(
    std::vector<dag_node_ptr>& ordered_nodes) const
{
  ordered_nodes.clear();

  std::vector<dag_node_ptr> nodes_to_process;

  _dag->for_each_node([&](dag_node_ptr n){
    nodes_to_process.push_back(n);
  });

  for(dag_node_ptr node : nodes_to_process)
    order_by_requirements_recursively(node, nodes_to_process, ordered_nodes);
}



}
}
}