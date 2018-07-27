#include "CL/sycl/context.hpp"
#include "CL/sycl/device.hpp"
#include "CL/sycl/queue.hpp"



namespace cl {
namespace sycl {
namespace detail {

stream_manager::stream_manager(const device& d)
{
  detail::set_device(d);
  detail::check_error(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
}

stream_manager::~stream_manager()
{
  detail::check_error(hipStreamDestroy(_stream));
}

hipStream_t stream_manager::get_stream() const
{
  return _stream;
}


}


queue::queue(const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{device{}}
{
  _stream = detail::stream_ptr(new detail::stream_manager{_device});
}

/// \todo constructors do not yet use asyncHandler
queue::queue(const async_handler &asyncHandler,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{device{}}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const device_selector &deviceSelector,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const device &syclDevice, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{syclDevice}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const device &syclDevice, const async_handler &asyncHandler,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{syclDevice}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


queue::queue(const context &syclContext, const device_selector &deviceSelector,
             const async_handler &asyncHandler, const property_list &propList)
  : detail::property_carrying_object{propList},
    _device{deviceSelector.select_device()}
{
  _stream = detail::stream_ptr{new detail::stream_manager{_device}};
}


context queue::get_context() const {
  return context{this->_device.get_platform()};
}

device queue::get_device() const {
  return this->_device;
}


bool queue::is_host() const {
  return false;
}

void queue::wait() {
  detail::check_error(hipStreamSynchronize(_stream->get_stream()));
}

void queue::wait_and_throw() {
  detail::check_error(hipStreamSynchronize(_stream->get_stream()));
}

void queue::throw_asynchronous() {}

bool queue::operator==(const queue& rhs) const
{ return (_device == rhs._device) && (_stream == rhs._stream); }

bool queue::operator!=(const queue& rhs) const
{ return !(*this == rhs); }


}// namespace sycl
}// namespace cl
