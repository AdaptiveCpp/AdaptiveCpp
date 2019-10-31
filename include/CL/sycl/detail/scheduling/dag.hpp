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

#include <vector>
#include <memory>

#include "hints.hpp"
#include "operations.hpp"
#include "dag_node.hpp"

namespace cl {
namespace sycl {
namespace detail {


class inorder_queue_execution_backend
{
public:
  /// \return the backend queue this object operates on
  virtual inorder_queue* get_queue() const = 0;

  /// Inserts an event into the stream
  virtual std::unique_ptr<dag_node_event> insert_event() = 0;

  virtual void submit_memcpy(const memcpy_operation&) = 0;
  virtual void submit_kernel(const kernel_operation&) = 0;
  virtual void submit_prefetch(const prefetch_operation&) = 0;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue may be from the same or a different backend.
  virtual void submit_queue_wait_for(dag_node_event*) = 0;

  virtual ~inorder_queue_execution_backend() {}
};

class dag;

class dag_expander
{
public:
  dag_expander(dag& d);

private:
  bool can_mem_requirements_be_merged(memory_requirement* a, memory_requirement* b) const;

  std::unordered_map<dag_node_ptr, std::vector<dag_node_ptr>> parents;
  // Maps a node that is scheduled for removal to all nodes having it as requirement
  std::vector<dag_node_ptr> nodes_for_removal;
  std::vector<dag_node_ptr> nodes_for_replacement;
  std::vector<std::vector<dag_node_ptr>> node_groups_for_merging;
};

class dag_scheduler
{
public:
  virtual cost_type estimate_runtime_cost() = 0;
  virtual void submit() = 0;
};


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
  // into actual operations or removed.
  void add_memory_requirement(dag_node_ptr requirement);

  const std::vector<dag_node_ptr>& get_kernels() const;
  
  bool is_executable() const { return _memory_requirements.empty(); }

  // Registers the dependencies of all requirements in this DAG with
  // the dependency trackers associated with the data buffers.
  // This function should only be called right before DAG execution,
  // after all optimization steps
  void commit_node_dependencies();

  using node_iterator = std::vector<dag_node_ptr>::iterator;

  bool contains_node(dag_node_ptr node) const;
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
}