#ifndef SYCU_EXCEPTION_HPP
#define SYCU_EXCEPTION_HPP

#include <stdexcept>
#include <exception>
#include <functional>

#include "types.hpp"
#include "backend/backend.hpp"
#include "context.hpp"

namespace cl {
namespace sycl {

using exception_ptr = exception_ptr_class;

class exception {
public:
  exception(const string_class &message, hipError_t error_code = hipErrorUnknown)
    : _msg{message}, _error_code{static_cast<int>(error_code)}
  {}

  const char *what() const
  {
    return _msg.c_str();
  }

  bool has_context() const
  {
    return false;
  }

  context get_context() const;

  /// Returns CUDA/HIP error codes
  int get_cl_code() const
  {
    return _error_code;
  }

private:
  string_class _msg;
  int _error_code;
};

using exception_list = vector_class<exception_ptr>;
using async_handler = function_class<void(cl::sycl::exception_list)>;


class unimplemented : public exception {
  using exception::exception;
};
class runtime_error : public exception {
  using exception::exception;
};
class kernel_error : public runtime_error {
  using runtime_error::runtime_error;
};
class accessor_error : public runtime_error {
  using runtime_error::runtime_error;
};
class nd_range_error : public runtime_error {
  using runtime_error::runtime_error;
};
class event_error : public runtime_error {
  using runtime_error::runtime_error;
};
class invalid_parameter_error : public runtime_error {
  using runtime_error::runtime_error;
};
class device_error : public exception {
  using exception::exception;
};
class compile_program_error : public device_error {
  using device_error::device_error;
};
class link_program_error : public device_error {
  using device_error::device_error;
};
class invalid_object_error : public device_error {
  using device_error::device_error;
};
class memory_allocation_error : public device_error {
  using device_error::device_error;
};
class platform_error : public device_error {
  using device_error::device_error;
};
class profiling_error : public device_error {
  using device_error::device_error;
};
class feature_not_supported : public device_error {
  using device_error::device_error;
};

namespace detail {

static void check_error(hipError_t e) {

  switch(e) {

  case hipSuccess:
    return;
  case hipErrorNotReady:
    // Not really an error, since it indicates that an operation is enqueued
    return;
  case hipErrorNoDevice:
    // This occurs when no devices are found. Do not throw in that case,
    // we want the user to treat the case when no devices are
    // available without try/catch
    return;
  case hipErrorInvalidContext:
    throw platform_error{"Input context is invalid", e};
  case hipErrorInvalidKernelFile:
    throw platform_error{"Invalid PTX",e};
  case hipErrorMemoryAllocation:
    throw memory_allocation_error{"Bad memory allocation",e};
  case hipErrorInitializationError:
    throw exception{"Initialization error", e};
  case hipErrorLaunchFailure:
    throw kernel_error{"An error occurred on the device while executing a kernel.", e};
  case hipErrorOutOfResources:
    throw device_error{"Out of resources", e};
  case hipErrorInvalidDevice:
    throw device_error{"Invalid device id", e};
  case hipErrorInvalidValue:
    throw runtime_error{"One or more of the parameters passed to the API "
                        "call is NULL or not in an acceptable range.",e};
  case hipErrorInvalidDevicePointer:
    throw invalid_parameter_error{"Invalid device pointer",e};
  case hipErrorInvalidMemcpyDirection:
    throw invalid_parameter_error{"Invalid memcpy direction",e};
  case hipErrorUnknown:
    throw exception{"Unknown HIP error",e};
  case hipErrorInvalidResourceHandle:
    throw invalid_object_error{"Invalid event or queue",e};
  case hipErrorRuntimeMemory:
    throw device_error{"HSA memory error",e};
  case hipErrorRuntimeOther:
    throw device_error{"HSA error",e};
  case hipErrorHostMemoryAlreadyRegistered:
    throw runtime_error{"Could not lock page-locked memory",e};
  case hipErrorHostMemoryNotRegistered:
    throw runtime_error{"Could not unlock non-page-locked memory",e};
  case hipErrorMapBufferObjectFailed:
    throw runtime_error{"IPC memory attach failed from ROCr",e};
  default:
    throw exception{"Unknown error occured", e};
  }

}

}

} // namespace sycl
} // namespace cl

#endif
