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
#include <cstddef>

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

      for(auto weak_node : _node->get_requirements()) {
        // TODO Is it correct to just use our handler here?
        if(auto node = weak_node.lock())
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
        _requires_runtime.get()->dag().flush_sync();
      
      assert(this->_node->is_submitted());
      this->_node->wait();
    }
  }

  static void wait(const vector_class<event> &eventList)
  {
    rt::runtime_keep_alive_token requires_runtime;
    // Only need a at most a single flush,
    // so check if any of the events are unsubmitted,
    // if so, perform a single flush.
    bool flush = false;
    for(const event& evt: eventList)
      if(evt._node)
        if(!evt._node->is_submitted())
          flush = true;

    if(flush)
      requires_runtime.get()->dag().flush_sync();

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
          [](exception_list e) { glue::default_async_handler(e); });
    else
      glue::throw_asynchronous_errors(eventList.front()._handler);
  }

  template <typename Param>
  typename Param::return_type get_info() const;

  // Wait for and retrieve profiling timestamps. Supports all handler operations except
  // handler::require() and handler::update_host().
  // Will throw sycl::invalid_object_error unless the queue was constructed with
  // `property::queue::enable_profiling`.
  // Timestamps are returned in std::chrono::system_clock nanoseconds-since-epoch.
  template <typename Param>
  typename Param::return_type get_profiling_info() const
  {
    if(!_node) {
      const auto error = rt::make_error(__acpp_here(),
                                        {"Operation not profiled: node is invalid.",
                                         rt::error_type::invalid_object_error});
      throw exception{make_error_code(errc::invalid), error.what()};
    }
    // Instrumentations are only set up once a node has passed
    // the scheduler.
    // This is needed to avoid a potential deadlock if the runtime
    // decides to queue up the operation to wait for more work,
    // but the user thread waits for the instrumentation results
    // and so cannot submit more work.
    if(!this->_node->is_submitted())
      _requires_runtime.get()->dag().flush_sync();

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
      const auto error = rt::make_error(__acpp_here(),
                                        {"Operation not profiled: "
                                         "Profiling was not requested by user.",
                                         rt::error_type::invalid_object_error});
      throw exception{make_error_code(errc::invalid), error.what()};
    }
    if(_node->get_operation()->is_requirement()) {
      const auto error = rt::make_error(__acpp_here(),
                                        {"Operation not profiled: "
                                         "hipSYCL currently does not support "
                                         "profiling explicit requirements "
                                         "or host updates",
                                         rt::error_type::invalid_object_error});
      throw exception{make_error_code(errc::invalid), error.what()};
    }

    if (std::is_same_v<Param, info::event_profiling::command_submit>) {
      auto submission =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::submission_timestamp>();
      
      if(!submission)
        throw exception{make_error_code(errc::invalid),                      
                        "Operation not profiled: No submission timestamp available"};

      return rt::profiler_clock::ns_ticks(submission->get_time_point());
    } else if (std::is_same_v<Param, info::event_profiling::command_start>) {
      auto start =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::execution_start_timestamp>();

      if(!start)
        throw exception{make_error_code(errc::invalid),
                        "Operation not profiled: No execution "
                        "start timestamp available"};

      return rt::profiler_clock::ns_ticks(start->get_time_point());
    } else if (std::is_same_v<Param, info::event_profiling::command_end>) {
      auto finish =
          _node->get_operation()
              ->get_instrumentations()
              .get<rt::instrumentations::execution_finish_timestamp>();
      
      if(!finish)
        throw exception{make_error_code(errc::invalid),
                        "Operation not profiled: No execution "
                        "end timestamp available"};

      return rt::profiler_clock::ns_ticks(finish->get_time_point());
    } else {
      throw exception{make_error_code(errc::invalid),
                      "Unknown event profiling request"};
    }
  }

  friend bool operator ==(const event& lhs, const event& rhs)
  { return lhs._node == rhs._node; }

  friend bool operator !=(const event& lhs, const event& rhs)
  { return !(lhs == rhs); }

  std::size_t AdaptiveCpp_hash_code() const {
    return std::hash<void*>{}(_node.get());
  }


  [[deprecated("Use AdaptiveCpp_hash_code()")]]
  auto hipSYCL_hash_code() const {
    return AdaptiveCpp_hash_code();
  }

private:

  rt::dag_node_ptr _node;
  rt::runtime_keep_alive_token _requires_runtime;
  async_handler _handler;
};

HIPSYCL_SPECIALIZE_GET_INFO(event, command_execution_status)
{
  if(!_node || _node->is_complete())
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

namespace std {

template <>
struct hash<hipsycl::sycl::event>
{
  std::size_t operator()(const hipsycl::sycl::event& e) const
  {
    return e.AdaptiveCpp_hash_code();
  }
};

}

#endif
