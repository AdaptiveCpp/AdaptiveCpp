/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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


#ifndef HIPSYCL_DAG_SUBMITTED_OPS_HPP
#define HIPSYCL_DAG_SUBMITTED_OPS_HPP

#include <mutex>
#include <vector>

#include "dag_node.hpp"
#include "generic/async_worker.hpp"

namespace hipsycl {
namespace rt {


class dag_submitted_ops
{
public:
  // Asynchronously waits on the nodes to complete, and, once complete,
  // removes them (and other completed) nodes from the submitted list.
  //
  // All nodes in the provided argument vector must have been registered
  // previously with update_with_submission()
  //
  // For best performance, the provided nodes should be in submission order.
  void async_wait_and_unregister(const std::vector<dag_node_ptr>& nodes);
  void update_with_submission(dag_node_ptr single_node);

  void wait_for_all();
  void wait_for_group(std::size_t node_group);
  std::vector<dag_node_ptr> get_group(std::size_t node_group);

  bool contains_node(dag_node_ptr node) const;
private:
  void purge_known_completed();

  std::vector<dag_node_ptr> _ops;
  mutable std::mutex _lock;
  worker_thread _updater_thread;
};


}
}

#endif
