/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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
  submit_directly(dag_node_ptr node, operation *op,
                  const node_list_t &reqs) override;

  inorder_queue* get_queue() const;

  bool can_execute_on_device(const device_id& dev) const override;
  bool is_submitted_by_me(dag_node_ptr node) const override;
private:
  std::unique_ptr<inorder_queue> _q;
  std::atomic<std::size_t> _num_submitted_operations;
};

}
}

#endif
