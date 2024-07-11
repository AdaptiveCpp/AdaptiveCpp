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
#ifndef HIPSYCL_INORDER_EXECUTOR_HPP
#define HIPSYCL_INORDER_EXECUTOR_HPP

#include <atomic>

#include "executor.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "inorder_queue.hpp"

namespace hipsycl {
namespace rt {

/// inorder_executor implements the executor
/// interface on top of inorder_queue objects.
///
/// This class is thread-safe, provided that the underlying
/// inorder_queue is thread-safe.
class inorder_executor : public backend_executor
{
public:
  inorder_executor(std::unique_ptr<inorder_queue> q);

  virtual ~inorder_executor();

  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;

  virtual void
  submit_directly(const dag_node_ptr& node, operation *op,
                  const node_list_t &reqs) override;

  inorder_queue* get_queue() const;

  bool can_execute_on_device(const device_id& dev) const override;
  bool is_submitted_by_me(const dag_node_ptr& node) const override;

  result wait();
private:
  std::unique_ptr<inorder_queue> _q;
  std::atomic<std::size_t> _num_submitted_operations;
};

}
}

#endif
