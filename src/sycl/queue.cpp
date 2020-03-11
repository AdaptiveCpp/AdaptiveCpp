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

#include "hipSYCL/sycl/context.hpp"
#include "hipSYCL/sycl/device.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/detail/application.hpp"


namespace hipsycl {
namespace sycl {
namespace detail {

stream_manager::stream_manager(const device& d)
  : stream_manager{d, [](exception_list){}}
{}

stream_manager::stream_manager(const device& d,
                               async_handler handler)
  : _dev{d},
    _handler{handler}
{
  detail::set_device(d);
  detail::check_error(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
}

stream_manager::stream_manager()
  : stream_manager{[](exception_list){}}
{}

stream_manager::stream_manager(async_handler handler)
  : _stream{0},
    _dev{},
    _handler{handler}
{}

stream_manager::~stream_manager()
{
  if(_stream != 0)
  {
    hipStreamDestroy(_stream);
  }
}

hipStream_t stream_manager::get_stream() const
{
  return _stream;
}

stream_ptr stream_manager::default_stream()
{
  return stream_ptr{new stream_manager()};
}

void stream_manager::activate_device() const
{
  detail::set_device(_dev);
}

async_handler stream_manager::get_error_handler() const
{
  return this->_handler;
}


device stream_manager::get_device() const
{
  return this->_dev; 
}

}


queue::queue(const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{device{}},
    _handler{[](exception_list){}}
{
  this->init();
}

/// \todo constructors do not yet use asyncHandler
queue::queue(const async_handler &asyncHandler,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{device{}},
    _handler{asyncHandler}
{
  this->init();
}


queue::queue(const device_selector &deviceSelector,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()},
    _handler{[](exception_list){}}
{
  this->init();
}


queue::queue(const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()},
    _handler{asyncHandler}
{
  this->init();
}


queue::queue(const device &syclDevice, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{syclDevice},
    _handler{[](exception_list){}}
{
  this->init();
}


queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{syclDevice},
    _handler{asyncHandler}
{
  this->init();
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()},
    _handler{[](exception_list){}}
{
  this->init();
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()},
    _handler{asyncHandler}
{
  this->init();
}

void queue::init()
{
  this->_stream = detail::stream_ptr{new detail::stream_manager{
      _device,
      _handler}};

  this->_hooks = detail::queue_submission_hooks_ptr{
        new detail::queue_submission_hooks{}};
}

context queue::get_context() const {
  return context{this->_device.get_platform()};
}

device queue::get_device() const {
  return this->_device;
}


bool queue::is_host() const {
  return get_device().is_host();
}

void queue::wait() {
  detail::application::get_task_graph().finish(_stream);
}

void queue::wait_and_throw() {
  detail::application::get_task_graph().finish(_stream);
}

void queue::throw_asynchronous() {}

bool queue::operator==(const queue& rhs) const
{ return (_device == rhs._device) && (_stream == rhs._stream); }

bool queue::operator!=(const queue& rhs) const
{ return !(*this == rhs); }

hipStream_t queue::get_hip_stream() const
{ return _stream->get_stream(); }

detail::stream_ptr queue::get_stream() const
{ return _stream; }

}// namespace sycl
}// namespace hipsycl
