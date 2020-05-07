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
  // Kernel execution
  void add_kernel(dag_node_ptr kernel);
  // Explicit copy operation - must be executed
  void add_memcpy(dag_node_ptr memcpy);
  // Explicit fill
  void add_fill(dag_node_ptr fill);
  // USM/SVM prefetch
  void add_prefetch(dag_node_ptr prefetch);
  // memory requirements. DAGs that have requirements are not
  // executable until all requirements have been translated
  // into actual operations or are removed.
  void add_memory_requirement(dag_node_ptr requirement);

  const std::vector<dag_node_ptr>& get_kernels() const;
  const std::vector<dag_node_ptr>& get_memory_requirements() const;
  
  bool is_executable() const { return _memory_requirements.empty(); }

  // Registers the dependencies of all requirements in this DAG with
  // the dependency trackers associated with the data buffers.
  // This function should only be called right before DAG execution,
  // after all optimization steps
  void commit_node_dependencies();

  using node_iterator = std::vector<dag_node_ptr>::iterator;

  bool contains_node(dag_node_ptr node) const;

  void for_each_node(std::function<void(dag_node_ptr)> handler) const;

  void dump(std::ostream& ostr) const;
private:
  void commit_dependencies(const std::vector<dag_node_ptr>& nodes);
  bool is_requirement_from_this_dag(const dag_node_ptr& node) const;

  std::vector<dag_node_ptr> _kernels;
  std::vector<dag_node_ptr> _memcopies;
  std::vector<dag_node_ptr> _fills;
  std::vector<dag_node_ptr> _prefetches;
  std::vector<dag_node_ptr> _memory_requirements;
};


}
}

#endif
