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
