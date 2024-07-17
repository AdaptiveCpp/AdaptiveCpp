/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include <limits>
#include <cassert>

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/generic/multi_event.hpp"

namespace hipsycl {
namespace rt {

dag_node::dag_node(const execution_hints &hints,
                   const node_list_t &requirements,
                   std::unique_ptr<operation> op,
                   runtime* rt)
    : _hints{hints},
      _assigned_executor{nullptr}, _event{nullptr}, _operation{std::move(op)},
      _is_submitted{false}, _is_complete{false}, _is_virtual{false},
      _is_cancelled{false}, _rt{rt} {
  
  for(const auto& req : requirements)
    _requirements.push_back(req);
}

dag_node::~dag_node() {}

bool dag_node::is_submitted() const { return _is_submitted; }

bool dag_node::is_complete() const {
  if (_is_complete)
    // If we already know that we are complete we don't
    // need to ask the event
    return true;
  if (!_is_submitted)
    // If we are not submitted yet, the event won't exist yet,
    // so prevent invalid accesses
    return false;

  // Remember if we are complete
  if (get_event()->is_complete()) {
    _is_complete = true;
  }
  return _is_complete;
}

bool dag_node::is_known_complete() const {
  return _is_complete;
}

bool dag_node::is_cancelled() const { return _is_cancelled; }

bool dag_node::is_virtual() const { return _is_virtual; }

void dag_node::mark_submitted(std::shared_ptr<dag_node_event> completion_evt)
{
  this->_event = std::move(completion_evt);
  this->_is_submitted = true;
}

void dag_node::mark_virtually_submitted()
{
  _is_virtual = true;
  std::vector<std::shared_ptr<dag_node_event>> events;
  for (auto req : get_requirements()) {
    if(auto r = req.lock()) {
      assert(r->is_submitted());
      events.push_back(r->get_event());
    }
  }
  mark_submitted(std::make_shared<dag_multi_node_event>(events));
}
    
void dag_node::cancel() {
  mark_virtually_submitted();
  this->_is_complete = true;
  this->_is_cancelled = true;
}

void dag_node::assign_to_executor(backend_executor *ctx)
{
  this->_assigned_executor = ctx;
}


void dag_node::assign_to_device(device_id dev) {
  this->_assigned_device = dev;
}

void dag_node::assign_to_execution_lane(
    void * lane_id) {
  this->_assigned_execution_lane = lane_id;
}

void dag_node::assign_execution_index(std::size_t index)
{
  this->_assigned_execution_index = index;
}

void dag_node::assign_effective_operation(std::unique_ptr<operation> op)
{
  assert(!_replacement_executed_operation);
  _replacement_executed_operation = std::move(op);
}

std::size_t dag_node::get_assigned_execution_index() const
{
  return this->_assigned_execution_index;
}

device_id dag_node::get_assigned_device() const { return _assigned_device; }

backend_executor *dag_node::get_assigned_executor() const
{
  return _assigned_executor;
}

void* dag_node::get_assigned_execution_lane() const
{
  return _assigned_execution_lane;
}

const execution_hints &dag_node::get_execution_hints() const { return _hints; }

execution_hints &dag_node::get_execution_hints() { return _hints; }

namespace {

template<class F>
void descend_requirement_tree(F&& f, const dag_node* n) {
  if(f(n)) {
    for(const auto& req : n->get_requirements()) {
      if(auto r = req.lock())
        descend_requirement_tree(f, r.get());
    }
  }
}

// Looks recursively in the requirement graph of current for x.
// Descends no more than current_level levels and does not
// descend into requirements that are known to be complete.
bool recursive_find(const dag_node_ptr &current, int current_level,
                    const dag_node_ptr &x) {
  if(!current)
    return false;
  if(current == x)
    return true;
  if(current_level <= 0)
    return false;

  for(const auto& req : current->get_requirements()) {
    if(auto r = req.lock()) {
      if(!r->is_known_complete()) {
        if(recursive_find(r, current_level - 1, x))
          return true;
      }
    }
  }
  return false;
}



}

// Add requirement if not already present
void dag_node::add_requirement(dag_node_ptr requirement)
{
  for (auto req : _requirements) {
    if (req.lock() == requirement)
      return;
  }

  auto is_reachable_from = [](dag_node_ptr from, dag_node_ptr to,
                              int max_levels) -> bool {
    return recursive_find(from, max_levels, to);
  };

  const int search_depth =
      application::get_settings().get<setting::dag_req_optimization_depth>();

  for(auto weak_existing_req : _requirements) {
    if(auto existing_req = weak_existing_req.lock()) {
      if(is_reachable_from(existing_req, requirement, search_depth)) {
        // The requirement is already reachable from an existing requirement,
        // inserting is unnecessary since the existing requirement
        // already provides sufficient synchronization.
        return;
      }
    }
  }

  // Remove requirements that are weaker than the new nequirement
  for(std::size_t i = 0; i < _requirements.size(); ++i) {
    if(auto r = _requirements[i].lock()) {
      if(is_reachable_from(requirement, r, search_depth)) {
        _requirements[i] = std::weak_ptr<dag_node>{};
      }
    }
  }
  _requirements.erase(std::remove_if(_requirements.begin(), _requirements.end(),
                                     [](std::weak_ptr<dag_node> weak_req) {
                                       return weak_req.expired();
                                     }),
                      _requirements.end());

  _requirements.push_back(requirement);
}

operation *dag_node::get_operation() const { return _operation.get(); }

const weak_node_list_t &dag_node::get_requirements() const
{
  return _requirements;
}

void dag_node::wait() const
{
  while (!_is_submitted);
  if(_is_complete)
    return;

  _event->wait();
  // All requirements are now also complete
  descend_requirement_tree([](const dag_node* current) -> bool{
    // Descend to all nodes that are not yet marked as complete,
    // so abort on nodes that are complete
    if(current->_is_complete)
      return false;
    // Otherwise mark as complete and return
    current->_is_complete = true;
    return true;
  }, this);

  _is_complete = true;
}

std::shared_ptr<dag_node_event>
dag_node::get_event() const{
  return _event;
}

void dag_node::for_each_nonvirtual_requirement(
    std::function<void(dag_node_ptr)> handler) const {
  
  if (is_known_complete())
    return;
  
  for (auto req : get_requirements()) {
    if(auto r = req.lock()) {
      if (!r->is_virtual()) {
        handler(r);
      } else {
        r->for_each_nonvirtual_requirement(handler);
      }
    } else {
      std::cout << "############## COULD NOT LOCK ##############\n";
    }
  }
}

runtime* dag_node::get_runtime() const {
  return _rt;
}

}
}
