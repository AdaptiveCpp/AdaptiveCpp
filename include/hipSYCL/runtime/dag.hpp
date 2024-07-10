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
#ifndef HIPSYCL_DAG_HPP
#define HIPSYCL_DAG_HPP

#include <vector>
#include <memory>
#include <functional>
#include <ostream>

#include "hints.hpp"
#include "operations.hpp"
#include "dag_node.hpp"

namespace hipsycl {
namespace rt {

/// Represents a DAG that can be transformed for execution (e.g., turning 
/// by requirements into actual operations).
///
/// Thread safety: None
class dag
{
public:
  void add_command_group(dag_node_ptr node);

  const node_list_t& get_command_groups() const;
  const node_list_t& get_memory_requirements() const;

  bool contains_node(dag_node_ptr node) const;

  void for_each_node(std::function<void(dag_node_ptr)> handler) const;

  std::size_t num_nodes() const
  {
    return _command_groups.size() + _memory_requirements.size();
  }

  bool is_requirement_from_this_dag(const dag_node_ptr &node) const;

  void dump(std::ostream& ostr) const;
private:
  node_list_t _command_groups;
  node_list_t _memory_requirements;
};


}
}

#endif
