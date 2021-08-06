/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_DAG_NODE_HPP
#define HIPSYCL_DAG_NODE_HPP

#include <memory>
#include <atomic>

#include "hints.hpp"
#include "event.hpp"


namespace hipsycl {
namespace rt {

class operation;
class backend_executor;

class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

class dag_node
{
public:
  dag_node(const execution_hints& hints,
          const std::vector<dag_node_ptr>& requirements,
          std::unique_ptr<operation> op);

  ~dag_node();

  bool is_submitted() const;
  bool is_complete() const;
  bool is_known_complete() const;
  bool is_cancelled() const;
  bool is_virtual() const;
  
  /// Only to be called by the backend executor/scheduler
  void mark_submitted(std::shared_ptr<dag_node_event> completion_evt);
  /// Only to be called by the backend executor/scheduler
  void mark_virtually_submitted();
  /// Only to be called by the backend executor/scheduler
  void cancel();
  /// Only to be called by the backend executor/scheduler
  void assign_to_executor(backend_executor* ctx);
  /// Only to be called by the backend executor/scheduler
  void assign_to_device(device_id dev);
  /// Only to be called by the backend executor/scheduler
  void assign_to_execution_lane(std::size_t lane_id);
  /// Can be used by the backend executor to store
  /// ordering information between nodes.
  /// Only to be called by the backend executor/scheduler
  void assign_execution_index(std::size_t index);
    /// Only to be called by the backend executor/scheduler.
  /// This API will be replaced once subnodes are available
  /// to implement multi-operation nodes.
  /// Sets an effective operation. The original operation will persist,
  /// but for_each_executed_operation() will recognize that this was
  /// the operation that was actually executed.
  void assign_effective_operation(std::unique_ptr<operation> op);

  device_id get_assigned_device() const;
  backend_executor *get_assigned_executor() const;
  std::size_t get_assigned_execution_lane() const;
  std::size_t get_assigned_execution_index() const;

  const execution_hints& get_execution_hints() const;
  execution_hints& get_execution_hints();

  // Add requirement if not already present
  void add_requirement(dag_node_ptr requirement);
  operation* get_operation() const;
  const std::vector<dag_node_ptr>& get_requirements() const;

  // Wait until the associated event has completed.
  // Can be invoked before the event has been set (pre-submission),
  // in which case the function will additionally wait
  // until the event exists.
  void wait() const;

  std::shared_ptr<dag_node_event> get_event() const;

  void for_each_nonvirtual_requirement(std::function<void(dag_node_ptr)>
                                           handler) const;
  /// Iterates across all operations that have actually been executed.
  /// Precondition: is_submitted() returns true
  template<class Handler>
  void for_each_executed_operation(Handler h) {
    assert(is_submitted());
    if(_replacement_executed_operation)
      h(_replacement_executed_operation.get());
    else
      h(_operation.get());
  }
private:
  execution_hints _hints;
  std::vector<dag_node_ptr> _requirements;

  device_id _assigned_device;
  backend_executor *_assigned_executor;
  std::size_t _assigned_execution_lane;
  std::size_t _assigned_execution_index;

  std::shared_ptr<dag_node_event> _event;
  std::unique_ptr<operation> _operation;
  /// This is a temporary solution to access operations
  /// executed for requirements; we should move to an
  /// API consisting of subnodes to properly handle
  /// dependencies for multi-operation cases
  std::unique_ptr<operation> _replacement_executed_operation;

  std::atomic<bool> _is_submitted;
  mutable std::atomic<bool> _is_complete;
  bool _is_virtual;
  std::atomic<bool> _is_cancelled;

};

}
}

#endif
