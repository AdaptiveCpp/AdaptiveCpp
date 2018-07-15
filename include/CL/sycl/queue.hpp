#ifndef SYCU_QUEUE_HPP
#define SYCU_QUEUE_HPP

#include "exception.hpp"
#include "types.hpp"
#include "property.hpp"
#include "backend/backend.hpp"
#include "device.hpp"
#include "context.hpp"
#include "event.hpp"
#include "handler.hpp"

namespace cl {
namespace sycl {
namespace detail {

class stream_manager
{
public:
  stream_manager(const device& d)
  {
    detail::set_device(d);
    detail::check_error(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
  }

  ~stream_manager()
  {
    detail::check_error(hipStreamDestroy(_stream));
  }

  hipStream_t get_stream() const
  {
    return _stream;
  }
private:

  hipStream_t _stream;
};

using stream_ptr = shared_ptr_class<stream_manager>;

}

class queue {

public:

  explicit queue(const property_list &propList = {})
    : _device{device{}}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }

  /// \todo constructors do not yet use property lists and asyncHandler
  queue(const async_handler &asyncHandler,
        const property_list &propList = {})
    : _device{device{}}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const device_selector &deviceSelector,
        const property_list &propList = {})
    : _device{deviceSelector.select_device()}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {})
    : _device{deviceSelector.select_device()}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const device &syclDevice, const property_list &propList = {})
    : _device{syclDevice}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const device &syclDevice, const async_handler &asyncHandler,
        const property_list &propList = {})
    : _device{syclDevice}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const context &syclContext, const device_selector &deviceSelector,
        const property_list &propList = {})
    : _device{deviceSelector.select_device()}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  queue(const context &syclContext, const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {})
    : _device{deviceSelector.select_device()}
  {
    _stream = stream_ptr{new stream_manager{_device}};
  }


  /* CL Interop is not supported
  queue(cl_command_queue clQueue, const context& syclContext,
        const async_handler &asyncHandler = {});
  */

  /* -- common interface members -- */


  /* -- property interface members -- */


  /* CL Interop is not supported
  cl_command_queue get() const;
  */

  context get_context() const {
    return context{this->_device.get_platform()};
  }

  device get_device() const {
    return this->_device;
  }


  bool is_host() const {
    return false;
  }


  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const {
    static_assert(false, "Unimplemented");
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


  void wait() {
    detail::check_error(hipStreamSynchronize(_stream->get_stream()));
  }

  void wait_and_throw() {
    detail::check_error(hipStreamSynchronize(_stream->get_stream()));
  }

  void throw_asynchronous() {}

private:
  device _device;
  detail::stream_ptr _stream;
};

template <> struct hash_class<cl::sycl::queue> {
  auto operator()(const cl::sycl::queue &q) const {
    // Forward the hashing to the implementation
    return q.hash();
  }

};

}// namespace sycl
}// namespace cl

#endif
