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

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/queue.hpp"

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/sycl/types.hpp"
#include <exception>

namespace hipsycl {
namespace sycl {


queue::queue(const property_list &propList)
  : detail::property_carrying_object{propList},
    _handler{[](exception_list){}}
{
  this->init();
}

/// \todo constructors do not yet use asyncHandler
queue::queue(const async_handler &asyncHandler,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _handler{asyncHandler}
{
  this->init();
}


queue::queue(const device_selector &deviceSelector,
             const property_list &propList)
    : detail::property_carrying_object{propList},
      _handler{[](exception_list) {}} {

  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      deviceSelector.select_device()._device_id));

  this->init();
}


queue::queue(const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
    : detail::property_carrying_object{propList}, _handler{asyncHandler} {
  
  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      deviceSelector.select_device()._device_id));
  
  this->init();
}


queue::queue(const device &syclDevice, const property_list &propList)
    : detail::property_carrying_object{propList}, _handler{
                                                      [](exception_list) {}} {

  
  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      syclDevice._device_id));

  this->init();
}


queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList)
    : detail::property_carrying_object{propList}, _handler{asyncHandler} {

  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      syclDevice._device_id));
  
  this->init();
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const property_list &propList)
    : detail::property_carrying_object{propList}, _handler{
                                                      [](exception_list) {}} {
  
  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      deviceSelector.select_device()._device_id));
  
  this->init();
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
    : detail::property_carrying_object{propList}, _handler{asyncHandler} {

  _default_hints.add_hint(rt::make_execution_hint<rt::hints::bind_to_device>(
      deviceSelector.select_device()._device_id));
  
  this->init();
}

void queue::init()
{
  this->_hooks = detail::queue_submission_hooks_ptr{
        new detail::queue_submission_hooks{}};
}

context queue::get_context() const {
  return context{get_device().get_platform()};
}

device queue::get_device() const {
  if (_default_hints.has_hint<rt::hints::bind_to_device>()) {
    rt::device_id id =
        _default_hints.get_hint<rt::hints::bind_to_device>()->get_device_id();
    return device{id};
  }
  return device{};
}


bool queue::is_host() const {
  return get_device().is_host();
}

void queue::wait() {
  rt::application::dag().flush_sync();
  rt::application::dag().wait();
}

void queue::wait_and_throw() {
  this->wait();
  this->throw_asynchronous();
}

void queue::throw_asynchronous() {
  sycl::exception_list exceptions;

  std::vector<rt::result> async_errors;
  rt::application::get_runtime().errors().pop_each_error(
      [&](const rt::result &err) {
        async_errors.push_back(err);
      });

  for(const auto& err : async_errors) {
    try {
      // TODO: Translate err into exception
    } catch (...) {
      exceptions.push_back(std::current_exception());
    }
  }

  if(!exceptions.empty())
    _handler(exceptions);
}


}// namespace sycl
}// namespace hipsycl
