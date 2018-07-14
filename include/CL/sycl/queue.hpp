#ifndef SYCU_QUEUE_HPP
#define SYCU_QUEUE_HPP

#include "exception.hpp"
#include "types.hpp"
#include "property.hpp"
#include "backend/backend.hpp"
#include "device.hpp"

namespace cl {
namespace sycl {
namespace detail {

class stream_manager
{
public:
  stream_manager()
  {
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
  {}

  queue(const async_handler &asyncHandler,
        const property_list &propList = {})
  {}


  queue(const device_selector &deviceSelector,
        const property_list &propList = {})
  {

  }


  queue(const device_selector &deviceSelector,
        const async_handler &asyncHandler, const property_list &propList = {});


  queue(const device &syclDevice, const property_list &propList = {});


  queue(const device &syclDevice, const async_handler &asyncHandler,
        const property_list &propList = {});


  queue(const context &syclContext, const device_selector &deviceSelector,

        const property_list &propList = {});


  queue(const context &syclContext, const device_selector &deviceSelector,

        const async_handler &asyncHandler, const property_list &propList = {});


  queue(cl_command_queue clQueue, const context& syclContext,

        const async_handler &asyncHandler = {});


  /* -- common interface members -- */


  /* -- property interface members -- */


  cl_command_queue get() const;

  context get_context() const;
  device get_device() const;


  bool is_host() const {
    return false;
  }


  template <info::queue param>
  typename info::param_traits<info::queue, param>::return_type get_info() const;


  template <typename T>
  event submit(T cgf);


  template <typename T>
  event submit(T cgf, const queue &secondaryQueue);


  void wait();


  void wait_and_throw();


  void throw_asynchronous();

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
