/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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


#ifndef SYCU_QUEUE_HPP
#define SYCU_QUEUE_HPP

#include "types.hpp"
#include "exception.hpp"

#include "property.hpp"
#include "backend/backend.hpp"
#include "device.hpp"
#include "device_selector.hpp"
#include "context.hpp"
#include "event.hpp"
#include "handler.hpp"
#include "info/queue.hpp"
#include "CL/sycl/property.hpp"

namespace cl {
namespace sycl {
namespace detail {


class stream_manager
{
public:
  stream_manager(const device& d);
  ~stream_manager();

  hipStream_t get_stream() const;
private:
  hipStream_t _stream;
};

using stream_ptr = shared_ptr_class<stream_manager>;

}

class queue : public detail::property_carrying_object
{
public:

  explicit queue(const property_list &propList = {});

  /// \todo constructors do not yet use asyncHandler
  queue(const async_handler &asyncHandler,
        const property_list &propList = {});

  queue(const device_selector &deviceSelector,
        const property_list &propList = {});

  queue(const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {});

  queue(const device &syclDevice, const property_list &propList = {});

  queue(const device &syclDevice, const async_handler &asyncHandler,
        const property_list &propList = {});

  queue(const context &syclContext, const device_selector &deviceSelector,
        const property_list &propList = {});

  queue(const context &syclContext, const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {});


  /* CL Interop is not supported
  queue(cl_command_queue clQueue, const context& syclContext,
        const async_handler &asyncHandler = {});
  */

  /* -- common interface members -- */


  /* -- property interface members -- */


  /* CL Interop is not supported
  cl_command_queue get() const;
  */

  context get_context() const;

  device get_device() const;

  bool is_host() const;


  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const {
    throw unimplemented{"queue::get_info() is unimplemented"};
  }


  template <typename T>
  event submit(T cgf) {
    detail::set_device(_device);

    handler cgh{*this};
    cgf(cgh);

    return detail::insert_event(_stream->get_stream());
  }

  template <typename T>
  event submit(T cgf, const queue &secondaryQueue) {
    detail::set_device(_device);

    try {
      handler cgh{*this};
      cgf(cgh);
      wait();
      return event();
    }
    catch(exception &e) {
      handler cgh{secondaryQueue};
      cgf(cgh);
      return detail::insert_event(secondaryQueue._stream->get_stream());
    }

  }


  void wait();

  void wait_and_throw();

  void throw_asynchronous();

  bool operator==(const queue& rhs) const;

  bool operator!=(const queue& rhs) const;
private:
  device _device;
  detail::stream_ptr _stream;
};

}// namespace sycl
}// namespace cl



#endif
