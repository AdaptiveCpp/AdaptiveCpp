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

#include "CL/sycl/detail/task_graph.hpp"
#include "CL/sycl/detail/debug.hpp"

#include <mutex>
#include <cassert>


namespace cl {
namespace sycl {
namespace detail {

void task_done_callback(hipStream_t stream,
                        hipError_t status,
                        void *userData)
{
  task_graph_node* node = reinterpret_cast<task_graph_node*>(userData);

  assert(node != nullptr);
  // The node should only be 'done' once the callback has been handled.
  assert(!node->is_done());

  HIPSYCL_DEBUG_INFO << "task_graph: node "
                     << node
                     << " has completed" << std::endl;

  try
  {
    node->get_graph()->invoke_async_submission(node, status);
  }
  catch (...)
  {
    // ToDo: Set error condition of node?
    HIPSYCL_DEBUG_ERROR << "task_graph: task_done_callback caught async error, "
                           " invoking async handler." << std::endl;

    exception_ptr e = std::current_exception();
    node->run_error_handler(sycl::exception_list{e});
  }

  node->_register_callback();
  assert(node->is_done());

}

task_graph_node::task_graph_node(task_functor tf,
                                 const vector_class<task_graph_node_ptr>& requirements,
                                 stream_ptr stream,
                                 async_handler error_handler,
                                 task_graph* tgraph)
  : _submitted{false},
    _callback_handled{false},
    _tf{tf},
    _requirements{requirements},
    _stream{stream},
    _handler{error_handler},
    _parent_graph{tgraph}
{}

bool task_graph_node::is_submitted() const
{
  return _submitted;
}

bool task_graph_node::is_ready() const
{
  for(const auto& requirement : _requirements)
    if(!requirement->is_done())
      return false;
  return true;
}

bool task_graph_node::is_done() const
{
  if(!_submitted)
    return false;

  assert(!_event.is_null_event());
  return _callback_handled;
}

void
task_graph_node::submit()
{
  assert(!_submitted);
  assert(is_ready());

  try
  {
    HIPSYCL_DEBUG_INFO << "task_graph: Submitting node "
                       << this << std::endl;

    _stream->activate_device();
    this->_event = _tf();
    _submitted = true;

    // ToDo: Only add callback if event is not already complete?

    detail::check_error(
          hipStreamAddCallback(_stream->get_stream(), task_done_callback,
                               reinterpret_cast<void*>(this), 0));

  }
  catch(...)
  {
    HIPSYCL_DEBUG_ERROR << "task_graph: submit() caught async error, "
                           " invoking async handler." << std::endl;
    // Submitted must be set to true to avoid
    // subsequent submissions
    _submitted = true;

    exception_ptr e = std::current_exception();
    _handler(sycl::exception_list{e});
  }

}

void
task_graph_node::_register_callback()
{
  this->_callback_handled = true;
}


task_graph*
task_graph_node::get_graph() const
{
  return _parent_graph;
}

void
task_graph_node::run_error_handler(sycl::exception_list e) const
{
  _handler(e);
}

detail::stream_ptr
task_graph_node::get_stream() const
{
  return _stream;
}

void
task_graph_node::wait()
{
  if(!_submitted)
  {
    for(auto& requirement : _requirements)
      requirement->wait();
  }
  // wait until submission - this shouldn't take long, once
  // all requirements are met
  while(!_submitted);

  assert(_submitted);
  this->_event.wait();

  // The callback should be executed immediately after
  // the event's completion
  while(!_callback_handled);

  assert(this->_event.is_done());
}

bool
task_graph_node::are_requirements_on_same_stream() const
{
  for(const auto& requirement : _requirements)
    if(requirement->get_stream()->get_stream() !=
       this->get_stream()->get_stream())
      return false;
  return true;
}

const vector_class<task_graph_node_ptr>&
task_graph_node::get_requirements() const
{
  return _requirements;
}

///////////////// task_graph /////////////////////////

task_graph::~task_graph()
{
  this->finish();
  assert(this->_worker.queue_size() == 0);
  for(const auto& node : _nodes)
    assert(node->is_done());
}

task_graph_node_ptr
task_graph::insert(task_functor tf,
                   const vector_class<task_graph_node_ptr>& requirements,
                   detail::stream_ptr stream,
                   async_handler handler)
{
  task_graph_node_ptr node = std::make_shared<task_graph_node>(tf,
                                                               requirements,
                                                               stream,
                                                               handler,
                                                               this);

  HIPSYCL_DEBUG_INFO << "task_graph: Receiving task node "
                     << node.get() << std::endl;
  HIPSYCL_DEBUG_INFO << "task_graph:  Dependencies: " << std::endl;
  for(const auto& req : requirements)
    HIPSYCL_DEBUG_INFO << "task_graph:    " << req.get() << std::endl;

  std::lock_guard<mutex_class> lock{_mutex};

  this->purge_finished_tasks();
  _nodes.push_back(node);

  // Trigger the on_task_completed handler to make sure
  // the task gets submitted if it is the first one
  this->invoke_async_submission(node.get(), hipSuccess);

  return node;
}

void
task_graph::purge_finished_tasks()
{
  for(auto it = _nodes.begin();
      it != _nodes.end();)
  {
    if((*it)->is_done())
      it = _nodes.erase(it);
    else
      ++it;
  }
}

void
task_graph::submit_eligible_tasks() const
{
  for(const auto& node : _nodes)
    if(!node->is_submitted() && node->is_ready())
    {
      node->submit();
      assert(node->is_submitted());
    }
}

void
task_graph::process_graph()
{
  std::lock_guard<mutex_class> lock{_mutex};
  // Don't call purge_finished_tasks() here for now
  // to avoid resource deallocation in the worker thread.
  // The implementation is not yet thread-safe enough
  // for multi-threaded destruction.
  this->submit_eligible_tasks();
}

void task_graph::flush()
{
  std::lock_guard<mutex_class> lock{_mutex};
  this->submit_eligible_tasks();
}

void task_graph::finish()
{

  // We work on a copy of the task graph; this allows us
  // to wait without locking the task graph the entire time
  vector_class<task_graph_node_ptr> nodes_snapshot;
  {
    std::lock_guard<mutex_class> lock{_mutex};

    // Update the task graph one last time
    this->purge_finished_tasks();
    this->submit_eligible_tasks();

    nodes_snapshot = _nodes;
  }
  // Wait until everything is submitted
  _worker.wait();

  for(auto& node : nodes_snapshot)
    node->wait();
}

void task_graph::finish(detail::stream_ptr stream)
{
  vector_class<task_graph_node_ptr> nodes_snapshot;
  {
    std::lock_guard<mutex_class> lock{_mutex};

    // Update the task graph one last time
    this->purge_finished_tasks();
    this->submit_eligible_tasks();

    for(const auto& node : _nodes)
      if(node->get_stream()->get_stream() == stream->get_stream())
        nodes_snapshot.push_back(node);
  }

  for(auto& node : nodes_snapshot)
    node->wait();
}

void task_graph::invoke_async_submission(task_graph_node* node,
                                         hipError_t status)
{
  // We need to offload the graph update/processing to a
  // separate worker thread because CUDA (and HIP?) forbid
  // calling functions that operate on streams in stream callbacks.

  _worker([node, status]()
  {
    try
    {
      detail::check_error(status);

      task_graph* graph = node->get_graph();
      graph->process_graph();
    }
    catch(...)
    {
      HIPSYCL_DEBUG_ERROR << "task_graph: task completion status returned error, "
                             " invoking async handler." << std::endl;
      exception_ptr e = std::current_exception();
      node->run_error_handler(sycl::exception_list{e});
    }
  });

}

}
}
}
