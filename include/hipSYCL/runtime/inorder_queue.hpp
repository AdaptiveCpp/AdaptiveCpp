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
#ifndef HIPSYCL_INORDER_QUEUE_HPP
#define HIPSYCL_INORDER_QUEUE_HPP

#include <memory>
#include <string>

#include "dag_node.hpp"
#include "hints.hpp"
#include "operations.hpp"
#include "error.hpp"
#include "code_object_invoker.hpp"

namespace hipsycl {
namespace rt {

class inorder_queue_status {
public:
  inorder_queue_status() = default;
  inorder_queue_status(bool is_complete)
  : _is_complete{is_complete} {}

  bool is_complete() const {
    return _is_complete;
  }

private:
  bool _is_complete;
};

/// Represents an in-order queue. Implementations of this abstract
/// interface have to be thread-safe.
class inorder_queue
{
public:

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() = 0;
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event() = 0;

  virtual result submit_memcpy(memcpy_operation&, const dag_node_ptr&) = 0;
  virtual result submit_kernel(kernel_operation&, const dag_node_ptr&) = 0;
  virtual result submit_prefetch(prefetch_operation &, const dag_node_ptr&) = 0;
  virtual result submit_memset(memset_operation&, const dag_node_ptr&) = 0;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(const dag_node_ptr& evt) = 0;
  virtual result submit_external_wait_for(const dag_node_ptr& node) = 0;

  virtual result wait() = 0;

  virtual device_id get_device() const = 0;
  /// Return native type if supported, nullptr otherwise
  virtual void* get_native_type() const = 0;

  virtual result query_status(inorder_queue_status& status) = 0;

  virtual ~inorder_queue(){}
};

}
}

#endif
