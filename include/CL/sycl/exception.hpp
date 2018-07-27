#ifndef SYCU_EXCEPTION_HPP
#define SYCU_EXCEPTION_HPP

#include <stdexcept>
#include <exception>
#include <functional>

#include "types.hpp"
#include "backend/backend.hpp"

namespace cl {
namespace sycl {

class context;

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

void check_error(hipError_t e);

}

} // namespace sycl
} // namespace cl

#endif
