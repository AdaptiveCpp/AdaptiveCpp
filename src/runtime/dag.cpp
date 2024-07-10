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
