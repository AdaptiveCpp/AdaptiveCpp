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

#include "CL/sycl/access.hpp"
#include "CL/sycl/detail/application.hpp"
#include "hipSYCL/runtime/dag_enumerator.hpp"
#include "hipSYCL/runtime/dag_expander.hpp"
#include "hipSYCL/runtime/data.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "CL/sycl/detail/debug.hpp"

namespace sycl = cl::sycl;

namespace hipsycl {
namespace rt {

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

    // We need to check here if the *pages* accessed by the memory
    // requirements overlap to correctly detect conflicts!
    auto page_range_a = buff_a->get_data_region()->get_page_range(
        buff_a->get_access_offset3d() * buff_a->get_element_size(),
        buff_a->get_access_range3d() * buff_a->get_element_size());

    auto page_range_b = buff_b->get_data_region()->get_page_range(
        buff_b->get_access_offset3d() * buff_b->get_element_size(),
        buff_b->get_access_range3d() * buff_b->get_element_size());
    
    if(!access_ranges_overlap(page_range_a.first,
                              page_range_a.second,
                              page_range_b.first,
                              page_range_b.second))
      return false;

    return true;
  }
  else {
    assert(false && "Non-buffer requirements are unimplemented");
  }

  return false;
}

std::size_t get_node_id(dag_node_ptr node)
{
  // We require that all nodes have been enumerated
  assert(node->has_node_id());

  return node->get_node_id();
}

device_id get_assigned_device(
    dag_node_ptr node,
    const std::vector<node_scheduling_annotation> &scheduling_info)
{
  std::size_t node_id = get_node_id(node);

  assert(node_id < scheduling_info.size());
  return scheduling_info[node_id].get_target_device();
}

/// Check if \c requirement has an equivalent memory access
/// (e.g., memory requirement) compared to \c other, such that
/// the memory access in \c requirement can be optimized away.
bool is_requirement_satisfied_by(
    dag_node_ptr requirement, dag_node_ptr other,
    const std::vector<node_scheduling_annotation> &scheduling_data) 
{
  if(is_memory_requirement(requirement) && is_memory_requirement(other)){
    if (get_assigned_device(requirement, scheduling_data) ==
        get_assigned_device(other, scheduling_data)) {
      

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
                               std::vector<dag_node_ptr>::const_iterator end,
                               const std::vector<node_scheduling_annotation>& scheduling_data)
{
  assert(is_memory_requirement(*begin));
  
  device_id begin_device = get_assigned_device(*begin, scheduling_data);

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

        device_id current_device = get_assigned_device(*it, scheduling_data);

        if(current_device != begin_device)
          return it;
      }
    }
    else {
      // Image requirement is still unimplemented
      assert(false && "Non-buffer memory requirements are unimplemented");
    }
  }

  return end;
}


/// Identifies requirements in \c ordered_nodes that can be merged and adds
/// marks them with a "forwarded-to" dag_expander annotation to indicate
/// that the scheduler should instead only take the node into account
/// that this node is forwarded to.
/// \param ordered_nodes A vector of nodes where the elements
/// are ordered such that all requirements precede the node in the vector for
/// all nodes in the vector.
void mark_mergeable_nodes(const std::vector<dag_node_ptr> &ordered_nodes,
                          const std::vector<node_scheduling_annotation>& scheduling_data,
                          dag_expansion_result& result)
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
          find_max_merge_candidate_range(mem_req_it, ordered_nodes.end(),
                                         scheduling_data);

      for(auto merge_candidate = mem_req_it;
          merge_candidate != merge_candidates_end; 
          ++merge_candidate){

        if(is_memory_requirement(*merge_candidate) &&
          processed_nodes.find(*merge_candidate) == processed_nodes.end()) {

          // If we can merge the node with the candidate,
          // mark the candidate as "forwarding" to the node

          // Check if the requirement of the candidate is also satisfied by mem_req_it
          if(is_requirement_satisfied_by(*merge_candidate, *mem_req_it, scheduling_data))
          {
            HIPSYCL_DEBUG_INFO << "dag_expander: Marking requirement "
                << mem_req_it->get() << " for merging with node "
                << merge_candidate->get() << std::endl;

            std::size_t candidate_id = get_node_id(*merge_candidate);

            result.node_annotations(candidate_id)
                .set_forward_to_node(*mem_req_it);
            
          }

        }
      }

      processed_nodes.insert(*mem_req_it);
    
    }
  }
}

/// Constructs a memcpy_operation using the memcpy hardware model under the
/// assumption that the update sources all point to rects of the same size.
std::unique_ptr<operation>
construct_memcpy(buffer_memory_requirement *mem_req, device_id target_device,
                 const std::vector<std::pair<device_id, range_store::rect>>
                     &update_sources) {

  for(auto elem : update_sources)
    if(elem.second.second != mem_req->get_access_range3d())
      assert(
          false &&
          "construct_memcpy(): Emitting multiple memory transfers for a single "
          "requirement is not supported.");

  memory_location target_location{target_device, mem_req->get_access_offset3d(),
                                  mem_req->get_data_region()};

  std::vector<memory_location> source_locations;
  for (const auto &source : update_sources)
    source_locations.push_back(memory_location{
        source.first, source.second.first, mem_req->get_data_region()});

  memory_location source_location = sycl::detail::application::get_hipsycl_runtime()
      .backends()
      .hardware_model()
      .get_memcpy_model()
      ->choose_source(source_locations, target_location,
                      mem_req->get_access_range3d());

  return make_operation<memcpy_operation>(source_location, target_location,
                                          mem_req->get_access_range3d());
}

} // anonymous namespace

dag_expander::dag_expander(const dag* d, const dag_enumerator& enumerator)
: _enumerator{enumerator}, _dag{d}
{
  // In the constructor, we linearize nodes to determine opportunities
  // for merging nodes. This data can be reused for subsequent
  // expansions.

  // This fills the ordered_nodes vector with
  // all nodes from the DAG in an order
  // that guarantees that an entry in the vector only depends
  // only on entries that precede it in the vector 
  this->order_by_requirements(this->_ordered_nodes);
 
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

void dag_expander::expand(
    const std::vector<node_scheduling_annotation> &scheduler_data,
    dag_expansion_result &out) const
{

  out.reset();

  // 1.) As a first step, we identify requirements that
  // can be merged into a single requirement.
  
  mark_mergeable_nodes(_ordered_nodes, scheduler_data, out);

  // Simulate memory accesses to determine if actual data transfers
  // are necessary and from which source device those accesses might come

  for (dag_node_ptr node : _ordered_nodes) {
    if (node->get_operation()->is_requirement()) {
      if (cast<requirement>(node->get_operation())->is_memory_requirement()) {
        if (cast<memory_requirement>(node->get_operation())
                ->is_buffer_requirement()) {
          
          buffer_memory_requirement *mem_req =
              cast<buffer_memory_requirement>(node->get_operation());

          // Find the right fork of this data region
          std::size_t data_region_id = mem_req->get_data_region()->get_id();

          if (out.memory_state(data_region_id) == nullptr)
            out.add_data_region_fork(
                data_region_id,
                std::move(mem_req->get_data_region()->create_fork()),
                mem_req->get_data_region().get());

          buffer_data_region *data = out.memory_state(data_region_id);

          device_id target_device = get_assigned_device(node, scheduler_data);

          // Make sure an allocation exists on the target device
          if (!data->has_allocation(target_device)) {
            buffer_data_region::allocation_function allocator =
                [target_device](sycl::range<3> num_elements,
                                std::size_t element_size) -> void * {
              
              // TODO: Find out optimal minimum alignment
              return sycl::detail::application::get_backend(target_device.get_backend())
                  .get_allocator(target_device)
                  ->allocate(128, num_elements.size() * element_size);
            };

            HIPSYCL_DEBUG_INFO
                << "dag_expander: Requesting new allocation on device "
                << target_device.get_id() << ", backend "
                << static_cast<int>(target_device.get_backend())
                << " for data region " << mem_req->get_data_region() << std::endl;
            
            data->add_placeholder_allocation(target_device, allocator);
          }

          // Look for outdated regions and potentiallly necessary data
          // transfers, if this node is not a forwarded node
          // (for forwarded nodes, data transfers will be handled the node
          // that it is forwarded to)
          std::size_t node_id = get_node_id(node);
          if(!out.node_annotations(node_id).is_node_forwarded()) {

            std::vector<range_store::rect> outdated_regions;
            data->get_outdated_regions(
                target_device, mem_req->get_access_offset3d(),
                mem_req->get_access_range3d(), outdated_regions);
            

            if (outdated_regions.empty()) {
              // Yay, no outdated regions, so we can just optimize this access
              // away!
              HIPSYCL_DEBUG_INFO
                  << "dag_expander: Optimizing away memory requirement "
                  << mem_req << std::endl;
              
              out.node_annotations(node_id).set_optimized_away();
            } else {
              // We need to convert this requirement into actual data transfers
              for (range_store::rect r : outdated_regions) {
                // Find candidates from which to update the data
                std::vector<std::pair<device_id, range_store::rect>> update_sources;
                data->get_update_source_candidates(target_device, r,
                                                   update_sources);

                out.node_annotations(node_id).add_replacement_operation(
                    construct_memcpy(mem_req, target_device, update_sources));
              }
            }
          }
          // In any case we need to update the data state:
          // * mark the range as valid on the target device
          // * mark the range as invalid on all other devices
          //   if it was a write access
          if (mem_req->get_access_mode() == sycl::access::mode::read) {
            data->mark_range_valid(target_device,
                                   mem_req->get_access_offset3d(),
                                   mem_req->get_access_range3d());
            
          } else {
            // This not only marks the range as valid, it marks the same range
            // on all other devices as invalid
            data->mark_range_current(target_device,
                                     mem_req->get_access_offset3d(),
                                     mem_req->get_access_range3d());
          }
          
        } else {
          assert(false && "Non-buffer requirements are unimplemented");
        }
        
      }
    }
  }

}


dag_expander_annotation::dag_expander_annotation()
    : _optimized_away{false}
{}

void dag_expander_annotation::set_optimized_away()
{
  _replacement_operations.clear();
  _forwarding_target = nullptr;
  _optimized_away = true;
}


void dag_expander_annotation::add_replacement_operation(
    std::unique_ptr<operation> op)
{
  _optimized_away = false;
  _forwarding_target = nullptr;
  _replacement_operations.push_back(std::move(op));
}

void dag_expander_annotation::set_forward_to_node(dag_node_ptr forward_to_node)
{
  _optimized_away = false;
  _replacement_operations.clear();
  _forwarding_target = forward_to_node;
}

bool dag_expander_annotation::is_optimized_away() const
{
  return _optimized_away;
}

bool dag_expander_annotation::is_operation_replaced() const
{
  return !_replacement_operations.empty();
}

bool dag_expander_annotation::is_node_forwarded() const
{
  return _forwarding_target != nullptr;
}

const std::vector<std::unique_ptr<operation>> &
dag_expander_annotation::get_replacement_operations() const
{
  return _replacement_operations;
}

dag_node_ptr dag_expander_annotation::get_forwarding_target() const
{
  return _forwarding_target;
}

dag_expansion_result::dag_expansion_result(
    const dag_enumerator &object_enumeration)
    : _num_nodes(object_enumeration.get_node_index_space_size()),
      _num_memory_buffers(object_enumeration.get_data_region_index_space_size())
{
  reset();
}

void dag_expansion_result::reset()
{
  _node_annotations = std::vector<dag_expander_annotation>(_num_nodes);
  _forked_memory_states.resize(_num_memory_buffers);
  _original_data_regions.resize(_num_memory_buffers);

  std::fill(_forked_memory_states.begin(), _forked_memory_states.end(), nullptr);
  std::fill(_original_data_regions.begin(), _original_data_regions.end(), nullptr);
}

dag_expander_annotation &
dag_expansion_result::node_annotations(std::size_t node_id)
{
  assert(node_id < _node_annotations.size());
  return _node_annotations[node_id];
}

const dag_expander_annotation &
dag_expansion_result::node_annotations(std::size_t node_id) const
{
  assert(node_id < _node_annotations.size());
  return _node_annotations[node_id];
}

buffer_data_region *
dag_expansion_result::memory_state(std::size_t data_region_id)
{
  assert(data_region_id < _forked_memory_states.size());
  return _forked_memory_states[data_region_id].get();
}

const buffer_data_region *
dag_expansion_result::memory_state(std::size_t data_region_id) const
{
  assert(data_region_id < _forked_memory_states.size());
  return _forked_memory_states[data_region_id].get();
}

void dag_expansion_result::add_data_region_fork(
    std::size_t data_region_id, std::unique_ptr<buffer_data_region> fork,
    buffer_data_region *original) 
{
  assert(data_region_id < _forked_memory_states.size());

  _forked_memory_states[data_region_id] = std::move(fork);
  _original_data_regions[data_region_id] = original;
}

buffer_data_region *
dag_expansion_result::original_data_region(std::size_t data_region_id)
{
  assert(data_region_id < _original_data_regions.size());
  return _original_data_regions[data_region_id];
}

const buffer_data_region *
dag_expansion_result::original_data_region(std::size_t data_region_id) const 
{
  assert(data_region_id < _original_data_regions.size());
  return _original_data_regions[data_region_id];
}

}
}
