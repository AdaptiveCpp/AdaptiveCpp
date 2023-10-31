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


#include <algorithm>
#include <cassert>

#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/dag.hpp"

namespace hipsycl {
namespace rt {

void dag::add_command_group(dag_node_ptr node) {
  for (auto weak_req : node->get_requirements()) {
    if(auto req = weak_req.lock()) {
      if (req->get_operation()->is_requirement())
        _memory_requirements.push_back(req);
    }
  }
  _command_groups.push_back(node);
}

const node_list_t &dag::get_command_groups() const {
  return _command_groups;
}

const node_list_t &dag::get_memory_requirements() const {
  return _memory_requirements;
}

bool dag::is_requirement_from_this_dag(const dag_node_ptr& node) const
{
  assert_is<requirement>(node->get_operation());

  return std::find(_memory_requirements.begin(), 
                  _memory_requirements.end(), node) 
                  != _memory_requirements.end();
}

bool dag::contains_node(dag_node_ptr node) const
{
  if(std::find(_command_groups.begin(), _command_groups.end(), node) != _command_groups.end())
    return true;
  if(std::find(_memory_requirements.begin(), _memory_requirements.end(), node) != _memory_requirements.end())
    return true;

  return false;
}

void dag::for_each_node(std::function<void(dag_node_ptr)> handler) const
{
  std::for_each(_command_groups.begin(), 
                _command_groups.end(), handler);
  std::for_each(_memory_requirements.begin(), 
                _memory_requirements.end(), handler);
}

}
}
