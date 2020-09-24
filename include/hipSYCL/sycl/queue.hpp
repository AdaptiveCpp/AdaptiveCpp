/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
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


#ifndef HIPSYCL_QUEUE_HPP
#define HIPSYCL_QUEUE_HPP

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/glue/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"

#include "types.hpp"
#include "exception.hpp"

#include "property.hpp"
#include "backend/backend.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "context.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "info/info.hpp"
#include "detail/function_set.hpp"

#include <exception>
#include <memory>
#include <mutex>

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename, int, access::mode, access::target>
class automatic_placeholder_requirement_impl;

using queue_submission_hooks =
  function_set<sycl::handler&>;
using queue_submission_hooks_ptr = 
  shared_ptr_class<queue_submission_hooks>;

}

namespace propert::queue {

class in_order : public detail::property
{};

}


class queue : public detail::property_carrying_object
{

  template<typename, int, access::mode, access::target>
  friend class detail::automatic_placeholder_requirement_impl;

public:
  explicit queue(const property_list &propList = {})
      : queue{default_selector{},
              [](exception_list e) { glue::default_async_handler(e); },
              propList} {
    assert(_default_hints.has_hint<rt::hints::bind_to_device>());
  }

  explicit queue(const async_handler &asyncHandler,
                 const property_list &propList = {})
      : queue{default_selector{}, asyncHandler, propList} {
    assert(_default_hints.has_hint<rt::hints::bind_to_device>());
  }

  explicit queue(const device_selector &deviceSelector,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList},
        _ctx{deviceSelector.select_device()} {

    _handler = _ctx._impl->handler;
    
    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        deviceSelector.select_device()._device_id));

    this->init();
  }

  explicit queue(const device_selector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList},
        _ctx{deviceSelector.select_device(), asyncHandler}, _handler{
                                                                asyncHandler} {

    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        deviceSelector.select_device()._device_id));

    this->init();
  }

  explicit queue(const device &syclDevice, const property_list &propList = {})
      : detail::property_carrying_object{propList}, _ctx{syclDevice} {

    _handler = _ctx._impl->handler;
    
    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        syclDevice._device_id));

    this->init();
  }

  explicit queue(const device &syclDevice, const async_handler &asyncHandler,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList},
        _ctx{syclDevice, asyncHandler}, _handler{asyncHandler} {

    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        syclDevice._device_id));

    this->init();
  }

  explicit queue(const context &syclContext,
                 const device_selector &deviceSelector,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList}, _ctx{syclContext} {

    _handler = _ctx._impl->handler;
    
    device dev = deviceSelector.select_device();

    if (!is_device_in_context(dev, syclContext))
      throw invalid_object_error{"queue: Device is not in context"};
    
    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        dev._device_id));

    this->init();
  }

  explicit queue(const context &syclContext,
                 const device_selector &deviceSelector,
                 const async_handler &asyncHandler,
                 const property_list &propList = {})
      : detail::property_carrying_object{propList}, _ctx{syclContext},
        _handler{asyncHandler} {

    device dev = deviceSelector.select_device();

    if (!is_device_in_context(dev, syclContext))
      throw invalid_object_error{"queue: Device is not in context"};

    _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
        dev._device_id));
    
    this->init();
  }

  ~queue() {
    this->throw_asynchronous();
  }


  context get_context() const {
    return _ctx;
  }

  device get_device() const {
    if (_default_hints.has_hint<rt::hints::bind_to_device>()) {
      rt::device_id id =
          _default_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();
      return device{id};
    }
    return device{};
  }

  bool is_host() const { return get_device().is_host(); }
  bool is_in_order() const {
    return _is_in_order;
  }

  void wait() {
    rt::application::dag().flush_sync();
    rt::application::dag().wait();
  }

  void wait_and_throw() {
    this->wait();
    this->throw_asynchronous();
  }

  void throw_asynchronous() {
    glue::throw_asynchronous_errors(_handler);
  }

  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;


  template <typename T>
  event submit(T cgf) {
    std::lock_guard<std::mutex> lock{*_lock};

    handler cgh{*this, _handler, _default_hints};
    
    this->get_hooks()->run_all(cgh);

    rt::dag_node_ptr node = execute_submission(cgf, cgh);
    
    return event{node, _handler};
  }

  template <typename T>
  event submit(T cgf, const queue &secondaryQueue) {
    try {

      size_t num_errors_begin =
          rt::application::get_runtime().errors().num_errors();

      event evt = submit(cgf);
      // Flush so that we see any errors during submission
      rt::application::dag().flush_sync();

      size_t num_errors_end =
          rt::application::get_runtime().errors().num_errors();

      bool submission_failed = false;
      // TODO This approach fails if an async handler has consumed
      // the errors in the meantime
      if(num_errors_end != num_errors_begin) {
        // Need to check if there was a kernel error..
        rt::application::get_runtime().errors().for_each_error(
            [&](const rt::result &err) {
              if (!err.is_success()) {
                if (err.info().get_error_type() ==
                    rt::error_type::kernel_error) {
                  submission_failed = true;
                }
              }
            });
      }

      if(!submission_failed) {
        return evt;
      } else {
        return secondaryQueue.submit(cgf);
      }
    }
    catch(exception&) {
      return secondaryQueue.submit(cgf);
    }

  }

  friend bool operator==(const queue& lhs, const queue& rhs)
  { return lhs._default_hints == rhs._default_hints; }

  friend bool operator!=(const queue& lhs, const queue& rhs)
  { return !(lhs == rhs); }

private:
  template <class Cgf>
  rt::dag_node_ptr execute_submission(Cgf cgf, handler &cgh) {
    if (is_in_order()) {
      auto previous = _previous_submission.lock();
      if(previous)
        cgh.depends_on(event{previous, _handler});
    }
    
    cgh(cgf);

    rt::dag_node_ptr node = this->extract_dag_node(cgh);
    if (is_in_order()) {
      _previous_submission = node;
    }
    return node;
  }
      
  bool is_device_in_context(const device &dev, const context &ctx) const {    
    std::vector<device> devices = ctx.get_devices();
    for (const auto context_dev : devices) {
      if (context_dev == dev)
        return true;
    }
    return false;
  }

  rt::dag_node_ptr extract_dag_node(sycl::handler& cgh) {
  
    const std::vector<rt::dag_node_ptr>& dag_nodes =
      cgh.get_cg_nodes();

    if(dag_nodes.empty()) {
      HIPSYCL_DEBUG_ERROR
          << "queue: Command queue evaluation did not result in the creation "
             "of events. Are there operations inside the command group?"
          << std::endl;
      return nullptr;
    }
    if(dag_nodes.size() > 1) {
      HIPSYCL_DEBUG_ERROR
          << "queue: Multiple events returned from command group evaluation; "
             "multiple operations in a single command group is not SYCL "
             "conformant. Returning event to the last operation"
          << std::endl;
    }
    return dag_nodes.back();
  }


  void init() {
    _is_in_order = this->has_property<property::queue::in_order>();
    _lock = std::make_shared<std::mutex>();

    this->_hooks = detail::queue_submission_hooks_ptr{
          new detail::queue_submission_hooks{}};
  }


  detail::queue_submission_hooks_ptr get_hooks() const
  {
    return _hooks;
  }
  
  detail::queue_submission_hooks_ptr _hooks;

  rt::execution_hints _default_hints;
  context _ctx;
  async_handler _handler;
  bool _is_in_order;

  std::weak_ptr<rt::dag_node> _previous_submission;
  std::shared_ptr<std::mutex> _lock;
};

HIPSYCL_SPECIALIZE_GET_INFO(queue, context)
{
  return get_context();
}

HIPSYCL_SPECIALIZE_GET_INFO(queue, device)
{
  return get_device();
}

HIPSYCL_SPECIALIZE_GET_INFO(queue, reference_count)
{
  return 1;
}



inline handler::handler(const queue &q, async_handler handler,
                 const rt::execution_hints &hints)
    : _queue{&q}, _local_mem_allocator{q.get_device()}, _handler{handler},
      _execution_hints{hints} {}


namespace detail{


template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
class automatic_placeholder_requirement_impl
{
public:
  automatic_placeholder_requirement_impl(sycl::queue &q, 
      sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                access::placeholder::true_t>* acc)
    : _acc{acc}, _is_required{false}, _hooks{q.get_hooks()}
  {
    acquire();
  }

  void reacquire()
  {
    if(!_is_required)
      acquire();
  }

  void release()
  {
    if(_is_required)
      _hooks->remove(_hook_id);
    _is_required = false;
  }

  ~automatic_placeholder_requirement_impl()
  {
    if(_is_required)
      release();
  }

  bool is_required() const { return _is_required; }
  
private:
  void acquire()
  {
    auto acc = _acc;
    _hook_id = _hooks->add([acc] (sycl::handler& cgh) mutable{
      cgh.require(*acc);
    });

    _is_required = true;
  }

  bool _is_required;

  sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                                  access::placeholder::true_t>* _acc;

  std::size_t _hook_id;
  detail::queue_submission_hooks_ptr _hooks;
};

}

namespace vendor {
namespace hipsycl {

template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
class automatic_placeholder_requirement
{
public:
  using impl_type = detail::automatic_placeholder_requirement_impl<
    dataT,dimensions,accessMode,accessTarget>;

  automatic_placeholder_requirement(queue &q, 
      accessor<dataT, dimensions, accessMode, accessTarget,
                access::placeholder::true_t>& acc)
  {
    _impl = std::make_unique<impl_type>(q, &acc);
  }

  automatic_placeholder_requirement(std::unique_ptr<impl_type> impl)
  : _impl{std::move(impl)}
  {}

  void reacquire()
  {
    _impl->reacquire();
  }

  void release()
  {
    _impl->release();
  }

  bool is_required() const
  {
    return _impl->is_required();
  }

private:
  std::unique_ptr<impl_type> _impl;
};

template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
inline auto automatic_require(queue &q, 
    accessor<dataT, dimensions, accessMode, accessTarget,access::placeholder::true_t>& acc)
{
  using requirement_type = automatic_placeholder_requirement<
    dataT, dimensions, accessMode, accessTarget>;

  using impl_type = typename requirement_type::impl_type;

  return requirement_type{std::make_unique<impl_type>(q, &acc)};
}

} // hipsycl
} // vendor



}// namespace sycl
}// namespace hipsycl



#endif
