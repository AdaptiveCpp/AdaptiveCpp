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

#ifndef HIPSYCL_EXCEPTION_HPP
#define HIPSYCL_EXCEPTION_HPP

#include <stdexcept>
#include <exception>
#include <functional>
#include <string>
#include <system_error>

#include "hipSYCL/runtime/error.hpp"
#include "types.hpp"
#include "libkernel/backend.hpp"

namespace hipsycl {
namespace sycl {

class context;

class exception {
public:
  exception(const std::string& message)
  : _msg{message}
  {
    set_error_code();
  }

  exception(const rt::result& details)
    : _msg{details.what()}, _error_details{details}
  {
    set_error_code();
  }

  const char *what() const
  {
    return _msg.c_str();
  }

  bool has_context() const
  {
    return false;
  }

  const std::error_code& code() const noexcept
  {
    return error_code;
  }

  // Implementation in context.hpp
  context get_context() const;

private:
  void set_error_code(){
    error_code = std::error_code(_error_details.info().error_code().get_code(), std::system_category());
  }
  std::error_code error_code;
  string_class _msg;
  rt::result _error_details;
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

} // namespace sycl
} // namespace hipsycl

#endif
