/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_TASK_GRAPH_HPP
#define HIPSYCL_TASK_GRAPH_HPP

#include "../types.hpp"
#include "../backend/backend.hpp"

#include "stream.hpp"
#include "util.hpp"
#include "async_worker.hpp"

#include <atomic>

namespace hipsycl {
namespace sycl {
namespace detail {

enum class task_state
{
  enqueued,
  complete
};

using task_functor = function_class<task_state ()>;

class task_graph_node;
using task_graph_node_ptr = shared_ptr_class<task_graph_node>;

class task_graph;

class task_graph_node
{
public:
  task_graph_node(task_functor tf,
                  const vector_class<task_graph_node_ptr>& requirements,
                  stream_ptr stream,
                  async_handler error_handler,
                  task_graph* tgraph);

  void wait();
  bool is_submitted() const;

  /// A node is considered 'done' if
  /// the enqueued HIP operation has completed
  /// and the task_graph has been notified of
  /// the completion.
  bool is_done() const;
  bool is_ready() const;
  bool are_requirements_on_same_stream() const;

  void submit();

  task_graph* get_graph() const;

  void run_error_handler(sycl::exception_list) const;

  detail::stream_ptr get_stream() const;

  void set_done();

  const vector_class<task_graph_node_ptr>& get_requirements() const;

  async_handler get_error_handler() const;
private:
  std::atomic<bool> _submitted;
  std::atomic<bool> _task_done;

  task_functor _tf;
  vector_class<task_graph_node_ptr> _requirements;
  spin_lock _requirements_lock; // prevent _requirements from being cleared while iterated over

  stream_ptr _stream;
  async_handler _handler;

  task_graph* _parent_graph;
};

class task_graph
{
public:
  task_graph_node_ptr insert(task_functor tf,
                             const vector_class<task_graph_node_ptr>& requirements,
                             detail::stream_ptr stream,
                             async_handler handler);

  void finish();
  void finish(detail::stream_ptr stream);

  /// Submits all available task nodes to hip streams,
  /// but does not wait until everything has finished.
  void flush();

  ~task_graph();

  /// A single processing step
  void process_graph();

  /// Handler that is executed when a task has finished.
  void invoke_async_submission(async_handler error_handler);
private:
  void purge_finished_tasks();
  void submit_eligible_tasks();

  vector_class<task_graph_node_ptr> _nodes;

  mutex_class _mutex;

  worker_thread _worker;

  std::atomic<int> _num_open_callbacks;
};

}
}
}

#endif
