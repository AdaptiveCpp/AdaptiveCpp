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

#ifndef HIPSYCL_EXECUTOR_HPP
#define HIPSYCL_EXECUTOR_HPP

#include "dag_node.hpp"
#include "operations.hpp"
#include "hints.hpp"


namespace cl {
namespace sycl {
namespace detail {

class dag_interpreter;
class dag_enumerator;
class node_scheduling_annotation;

struct backend_execution_lane_range
{
  std::size_t begin;
  std::size_t num_lanes;
};

class backend_executor
{
public:

  virtual bool is_inorder_queue() const = 0;
  virtual bool is_outoforder_queue() const = 0;
  virtual bool is_taskgraph() const = 0;

  // The range of lanes to use per device
  virtual backend_execution_lane_range
  get_memcpy_execution_lane_range() const = 0;

  // The range of lanes to use per device
  virtual backend_execution_lane_range
  get_kernel_execution_lane_range() const = 0;

  virtual void
  submit_dag(const dag_interpreter &interpreter,
             const dag_enumerator &enumerator,
             const std::vector<node_scheduling_annotation> &annotations) = 0;

  // The create_event_* functions will typically be called
  // * by the scheduler, to implement features such as profiling;
  // * by this (or another) backend_executor in order to implement
  //   the create_wait_* functions since they typically require events
  //   after the node they wait for
  virtual std::unique_ptr<event_before_node>
  create_event_before(dag_node_ptr node) const = 0;

  virtual std::unique_ptr<event_before_node>
  create_event_after(dag_node_ptr node) const = 0;

  // The create_wait_* functions will be called by the scheduler to mark
  // synchronization points
  virtual std::unique_ptr<wait_for_node_on_same_lane>
  create_wait_for_node_same_lane(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const = 0;

  virtual std::unique_ptr<wait_for_node_on_same_backend>
  create_wait_for_node_same_backend(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const = 0;

  virtual std::unique_ptr<wait_for_external_node> 
  create_wait_for_external_node(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const = 0;

  virtual ~backend_executor(){}
};

class multi_inorder_queue_executor : public backend_executor
{
public:
  virtual ~multi_inorder_queue_executor() {}

  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;

  // The range of lanes to use per device
  backend_execution_lane_range
  get_memcpy_execution_lane_range() const override;

  // The range of lanes to use per device
  backend_execution_lane_range
  get_kernel_execution_lane_range() const override;

  void
  submit_dag(const dag_interpreter &interpreter,
             const dag_enumerator &enumerator,
             const std::vector<node_scheduling_annotation> &annotations) override;

  // The create_event_* functions will typically be called
  // * by the scheduler, to implement features such as profiling;
  // * by this (or another) backend_executor in order to implement
  //   the create_wait_* functions since they typically require events
  //   after the node they wait for
  virtual std::unique_ptr<event_before_node>
  create_event_before(dag_node_ptr node) const override;

  virtual std::unique_ptr<event_before_node>
  create_event_after(dag_node_ptr node) const override;

  // The create_wait_* functions will be called by the scheduler to mark
  // synchronization points
  std::unique_ptr<wait_for_node_on_same_lane>
  create_wait_for_node_same_lane(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const override;

  std::unique_ptr<wait_for_node_on_same_backend>
  create_wait_for_node_same_backend(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const override;

  std::unique_ptr<wait_for_external_node> 
  create_wait_for_external_node(
      dag_node_ptr node, const node_scheduling_annotation &annotation,
      dag_node_ptr other,
      node_scheduling_annotation &other_annotation) const override;
};

class inorder_queue : public backend_executor
{
public:
  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;


  /// Inserts an event into the stream
  virtual std::unique_ptr<dag_node_event> insert_event() = 0;

  virtual void submit_memcpy(const memcpy_operation&) = 0;
  virtual void submit_kernel(const kernel_operation&) = 0;
  virtual void submit_prefetch(const prefetch_operation&) = 0;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue may be from the same or a different backend.
  virtual void submit_queue_wait_for(dag_node_event*) = 0;
};


}
}
}

#endif
