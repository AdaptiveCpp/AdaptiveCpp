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

  const std::vector<dag_node_ptr>& get_command_groups() const;
  const std::vector<dag_node_ptr>& get_memory_requirements() const;

  using node_iterator = std::vector<dag_node_ptr>::iterator;

  bool contains_node(dag_node_ptr node) const;

  void for_each_node(std::function<void(dag_node_ptr)> handler) const;

  std::size_t num_nodes() const
  {
    return _command_groups.size() + _memory_requirements.size();
  }

  bool is_requirement_from_this_dag(const dag_node_ptr &node) const;

  void dump(std::ostream& ostr) const;
private:
  std::vector<dag_node_ptr> _command_groups;
  std::vector<dag_node_ptr> _memory_requirements;
};


}
}

#endif
