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
#include "detail/stream.hpp"
#include "detail/function_set.hpp"


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


class queue : public detail::property_carrying_object
{

  template<typename, int, access::mode, access::target>
  friend class detail::automatic_placeholder_requirement_impl;

public:

  explicit queue(const property_list &propList = {});

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
  typename info::param_traits<info::queue, param>::return_type get_info() const;


  template <typename T>
  event submit(T cgf) {
    _stream->activate_device();

    handler cgh{*this, _handler};

    this->get_hooks()->run_all(cgh);

    cgf(cgh);

    event evt = cgh._detail_get_event();

    return evt;
  }

  template <typename T>
  event submit(T cgf, const queue &secondaryQueue) {
    _stream->activate_device();

    try {
      handler cgh{*this, _handler};

      this->get_hooks()->run_all(cgh);

      cgf(cgh);

      // We need to wait to make sure everything is fine.
      // ToDo: Check for asynchronous errors.
      wait();
      return event();
    }
    catch(exception&) {
      return secondaryQueue.submit(cgf);
    }

  }


  void wait();

  /// \todo implement these properly
  void wait_and_throw();

  void throw_asynchronous();

  friend bool operator==(const queue& lhs, const queue& rhs)
  { return (lhs._device == rhs._device) && (lhs._stream == rhs._stream); }

  friend bool operator!=(const queue& lhs, const queue& rhs)
  { return !(lhs == rhs); }

  hipStream_t get_hip_stream() const;
  detail::stream_ptr get_stream() const;

private:

  void init();

  detail::queue_submission_hooks_ptr get_hooks() const
  {
    return _hooks;
  }

  device _device;
  detail::stream_ptr _stream;
  async_handler _handler;
  detail::queue_submission_hooks_ptr _hooks;

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
  return _stream.use_count();
}




namespace detail{


template<typename dataT, int dimensions, access::mode accessMode,
            access::target accessTarget>
class automatic_placeholder_requirement_impl
{
public:
  automatic_placeholder_requirement_impl(sycl::queue &q, 
      sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                access::placeholder::true_t> acc)
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

  bool is_required() const
  {
    return _is_required;
  }
private:
  void acquire()
  {
    auto acc = _acc;
    _hook_id = _hooks->add([acc](sycl::handler& cgh){
      cgh.require(acc);
    });

    _is_required = true;
  }

  bool _is_required;

  sycl::accessor<dataT, dimensions, accessMode, accessTarget,
                                  access::placeholder::true_t> _acc;

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
                access::placeholder::true_t> acc)
  {
    _impl = std::make_unique<impl_type>(q, acc);
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
    accessor<dataT, dimensions, accessMode, accessTarget,access::placeholder::true_t> acc)
{
  using requirement_type = automatic_placeholder_requirement<
    dataT, dimensions, accessMode, accessTarget>;

  using impl_type = typename requirement_type::impl_type;

  return requirement_type{std::make_unique<impl_type>(q, acc)};
}

} // hipsycl
} // vendor



}// namespace sycl
}// namespace hipsycl



#endif
