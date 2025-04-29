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
#ifndef HIPSYCL_DAG_SUBMITTED_OPS_HPP
#define HIPSYCL_DAG_SUBMITTED_OPS_HPP

#include <mutex>
#include <vector>

#include "dag_node.hpp"
#include "generic/async_worker.hpp"
#include "hints.hpp"

namespace hipsycl {
namespace rt {


class dag_submitted_ops
{
public:
  // Asynchronously waits on the nodes to complete, and, once complete,
  // removes them (and other completed) nodes from the submitted list.
  void async_wait_and_unregister();
  void update_with_submission(dag_node_ptr single_node);
  
  void wait_for_all();
  void wait_for_group(std::size_t node_group);
  node_list_t get_group(std::size_t node_group);

  bool contains_node(dag_node_ptr node) const;

  std::size_t get_num_nodes() const;

  ~dag_submitted_ops();
private:
  void purge_known_completed();
  void copy_node_list(std::vector<dag_node_ptr>& out) const;

  std::vector<dag_node_ptr> _ops;
  mutable std::mutex _lock;
  worker_thread _updater_thread;
};


}
}

#endif
