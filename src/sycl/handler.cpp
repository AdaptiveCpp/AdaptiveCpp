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

#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/queue.hpp"

namespace hipsycl {
namespace sycl {

handler::handler(const queue& q, async_handler handler)
: _queue{&q},
  _local_mem_allocator{q.get_device()},
  _handler{handler}
{}

hipStream_t handler::get_hip_stream() const
{
  return _queue->get_hip_stream();
}

detail::stream_ptr handler::get_stream() const
{
  return _queue->get_stream();
}

void handler::select_device() const
{
  detail::set_device(this->_queue->get_device());
}

detail::task_graph_node_ptr handler::submit_task(detail::task_functor f)
{
  auto& task_graph = detail::application::get_task_graph();

  bool enable_profiling = _queue->has_property<property::queue::enable_profiling>();
  auto graph_node =
      task_graph.insert(f, _spawned_task_nodes, get_stream(), enable_profiling, _handler);

  // Add new node to the access log of buffers. This guarantees that
  // subsequent buffer accesses will wait for existing tasks to complete,
  // if necessary
  for(const auto& buffer_access : _accessed_buffers)
  {
    buffer_access.buff->register_external_access(
        graph_node,
        buffer_access.access_mode);
  }

  _spawned_task_nodes.push_back(graph_node);
  return graph_node;
}

} // sycl
} // hipsycl
