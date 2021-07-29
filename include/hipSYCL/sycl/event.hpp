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


#ifndef HIPSYCL_EVENT_HPP
#define HIPSYCL_EVENT_HPP

#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/sycl/info/event.hpp"
#include "types.hpp"
#include "libkernel/backend.hpp"
#include "exception.hpp"
#include "info/info.hpp"

#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/instrumentation.hpp"

namespace hipsycl {
namespace sycl {

class event {
  friend class handler;
public:
  event()
  {}

  event(
      const rt::dag_node_ptr &evt,
      async_handler handler =
          [](exception_list e) { glue::default_async_handler(e); })
      : _node{evt} {}

  std::vector<event> get_wait_list()
  {
    if(_node) {
      std::vector<event> events;

      for(auto node : _node->get_requirements()) {
        // TODO Is it correct to just use our handler here?
        events.push_back(event{node, _handler});
      }

      return events;
      
    }
    return std::vector<event>{};
  }

  void wait()
  {
    if(this->_node){
      if(!this->_node->is_submitted())
        rt::application::dag().flush_sync();
      
      assert(this->_node->is_submitted());
      this->_node->wait();
    }
  }

  static void wait(const vector_class<event> &eventList)
  {
    // Only need a at most a single flush,
    // so check if any of the events are unsubmitted,
    // if so, perform a single flush.
    bool flush = false;
    for(const event& evt: eventList)
      if(evt._node)
        if(!evt._node->is_submitted())
          flush = true;

    if(flush)
      rt::application::dag().flush_sync();

    for(const event& evt: eventList){
      const_cast<event&>(evt).wait();
    }
  }

  void wait_and_throw()
  {
    wait();
    glue::throw_asynchronous_errors(_handler);
  }

  static void wait_and_throw(const vector_class<event> &eventList)
  {
    wait(eventList);

    // Just invoke handler of first event?
    if(eventList.empty())
      glue::throw_asynchronous_errors(
          [](sycl::exception_list e) { glue::default_async_handler(e); });
    else
      glue::throw_asynchronous_errors(eventList.front()._handler);
  }

  template <info::event param>
  typename info::param_traits<info::event, param>::return_type get_info() const;

  // Wait for and retrieve profiling timestamps. Supports all handler operations except
  // handler::require() and handler::update_host().
  // Will throw sycl::invalid_object_error unless the queue was constructed with
  // `property::queue::enable_profiling`.
  // Timestamps are returned in std::chrono::system_clock nanoseconds-since-epoch.
  template <info::event_profiling param>
  typename info::param_traits<info::event_profiling, param>::return_type get_profiling_info() const
  {
    if(!_node) {
      throw invalid_object_error{rt::make_error(
          __hipsycl_here(),
          {"Operation not profiled: node is invalid.", rt::error_type::invalid_object_error})};
    }
    // Instrumentations are only set up once a node has passed
    // the scheduler.
    // This is needed to avoid a potential deadlock if the runtime
    // decides to queue up the operation to wait for more work,
    // but the user thread waits for the instrumentation results
    // and so cannot submit more work.
    if(!this->_node->is_submitted())
      rt::application::dag().flush_sync();

    rt::execution_hints& hints = _node->get_execution_hints();
    // The regular SYCL API will always result in full profiling requested,
    // so we can check for the presence of all three timestamp instrumentation
    // requests.
    bool was_full_profiling_requested =
        hints.has_hint<
            rt::hints::request_instrumentation_submission_timestamp>() &&
        hints.has_hint<rt::hints::request_instrumentation_start_timestamp>() &&
        hints.has_hint<rt::hints::request_instrumentation_finish_timestamp>();

    if(!was_full_profiling_requested) {
      throw invalid_object_error{rt::make_error(
          __hipsycl_here(),
          {"Operation not profiled: Profiling was not requested by user.", rt::error_type::invalid_object_error})};
    }
    if(_node->get_operation()->is_requirement()) {
      throw invalid_object_error{rt::make_error(
          __hipsycl_here(),
          {"Operation not profiled: hipSYCL currently does not support "
           "profiling explicit requirements or host updates",
           rt::error_type::invalid_object_error})};
    }

    if (param == info::event_profiling::command_submit) {
      auto submission =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::submission_timestamp>();
      
      if(!submission)
          throw invalid_object_error(
              "Operation not profiled: No submission timestamp available");

      return rt::profiler_clock::ns_ticks(submission->get_time_point());
    } else if (param == info::event_profiling::command_start) {
      auto start =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::execution_start_timestamp>();

      if(!start)
          throw invalid_object_error(
              "Operation not profiled: No execution start timestamp available");

      return rt::profiler_clock::ns_ticks(start->get_time_point());
    } else if (param == info::event_profiling::command_end) {
      auto finish =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::execution_finish_timestamp>();
      
      if(!finish)
          throw invalid_object_error(
              "Operation not profiled: No execution end timestamp available");

      return rt::profiler_clock::ns_ticks(finish->get_time_point());
    } else {
      throw invalid_parameter_error{"Unknown event profiling request"};
    }
  }

  friend bool operator ==(const event& lhs, const event& rhs)
  { return lhs._node == rhs._node; }

  friend bool operator !=(const event& lhs, const event& rhs)
  { return !(lhs == rhs); }

private:

  rt::dag_node_ptr _node;
  async_handler _handler;
};

HIPSYCL_SPECIALIZE_GET_INFO(event, command_execution_status)
{
  if(_node->is_complete())
    return info::event_command_status::complete;

  if(_node->is_submitted())
    return info::event_command_status::running;

  return info::event_command_status::submitted;
}

HIPSYCL_SPECIALIZE_GET_INFO(event, reference_count)
{
  return _node.use_count();
}


} // namespace sycl
} // namespace hipsycl

#endif
