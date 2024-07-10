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
#ifndef HIPSYCL_DAG_NODE_HPP
#define HIPSYCL_DAG_NODE_HPP

#include <memory>
#include <atomic>

#include "hints.hpp"
#include "event.hpp"
#include "hipSYCL/common/small_vector.hpp"


namespace hipsycl {
namespace rt {

class operation;
class backend_executor;

class dag_node;
class runtime;
// These two aliases should be consistent with the definitions in
// operations.hpp, where they are defined as well.
// TODO: Unify these two separate alias definitions!
using dag_node_ptr = std::shared_ptr<dag_node>;
using node_list_t = common::small_vector<dag_node_ptr, 8>;
using weak_node_list_t = common::small_vector<std::weak_ptr<dag_node>, 8>;

class dag_node
{
public:
  dag_node(const execution_hints& hints,
          const node_list_t& requirements,
          std::unique_ptr<operation> op,
          runtime* rt);

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
  void assign_to_execution_lane(void* lane);
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
  // Returns potential additional information about execution lane
  // maintained by the backend executor.
  void* get_assigned_execution_lane() const;
  std::size_t get_assigned_execution_index() const;

  const execution_hints& get_execution_hints() const;
  execution_hints& get_execution_hints();

  // Add requirement if not already present
  void add_requirement(dag_node_ptr requirement);
  operation* get_operation() const;
  const weak_node_list_t& get_requirements() const;

  // Wait until the associated event has completed.
  // Can be invoked before the event has been set (pre-submission),
  // in which case the function will additionally wait
  // until the event exists.
  //
  // Waiting will cause the node and all its requirements to return true
  // for is_known_complete().
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

  runtime* get_runtime() const;
private:
  execution_hints _hints;
  weak_node_list_t _requirements;

  device_id _assigned_device;
  backend_executor *_assigned_executor;
  void* _assigned_execution_lane;
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

  runtime* _rt;

};

}
}

#endif
