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
#include <cassert>

#include "CL/sycl/detail/scheduling/dag_node.hpp"
#include "CL/sycl/detail/scheduling/operations.hpp"

namespace cl {
namespace sycl {
namespace detail {

dag_node::dag_node(const execution_hints &hints,
                   const std::vector<dag_node_ptr> &requirements,
                   std::unique_ptr<operation> op)
    : _hints{hints}, _requirements{requirements}, _assigned_executor{nullptr},
      _event{nullptr}, _operation{std::move(op)}, _is_submitted{false},
      _is_complete{false}, _node_id{std::numeric_limits<std::size_t>::max()} {}

bool dag_node::is_submitted() const { return _is_submitted; }

bool dag_node::is_complete() const { return _is_complete; }

void dag_node::mark_submitted(std::unique_ptr<dag_node_event> completion_evt)
{
  this->_event = std::move(completion_evt);
  this->_is_submitted = true;
}

void dag_node::assign_to_executor(backend_executor *ctx)
{
  this->_assigned_executor = ctx;
}


void dag_node::assign_to_device(device_id dev)
{
  this->_assigned_device = dev;
}

void dag_node::assign_to_execution_lane(std::size_t lane_id)
{
  this->_assigned_execution_lane = lane_id;
}

device_id dag_node::get_assigned_device() const { return _assigned_device; }

backend_executor *dag_node::get_assigned_executor() const
{
  return _assigned_executor;
}

std::size_t dag_node::get_assigned_execution_lane() const
{
  return _assigned_execution_lane;
}

const execution_hints &dag_node::get_execution_hints() const { return _hints; }

execution_hints &dag_node::get_execution_hints() { return _hints; }

// Add requirement if not already present
void dag_node::add_requirement(dag_node_ptr requirement)
{
  for (auto req : _requirements) {
    if (req == requirement)
      return;
  }
  _requirements.push_back(requirement);
}

operation *dag_node::get_operation() const { return _operation.get(); }

const std::vector<dag_node_ptr> &dag_node::get_requirements() const
{
  return _requirements;
}

void dag_node::wait() const
{
  while (!_is_submitted);

  _event->wait();
}

void dag_node::assign_node_id(std::size_t id) {
  assert(!has_node_id());

  _node_id = id;
}

bool dag_node::has_node_id() const{
  return _node_id == std::numeric_limits<std::size_t>::max();
}

std::size_t dag_node::get_node_id() const {
  assert(has_node_id());
  return _node_id;
}

}
}
}