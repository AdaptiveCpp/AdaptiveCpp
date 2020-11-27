/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_RT_ERROR_HPP
#define HIPSYCL_RT_ERROR_HPP

#include <functional>
#include <ostream>
#include <string>
#include <sstream>
#include <memory>

#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

enum class error_type {
  unimplemented,
  runtime_error,
  kernel_error,
  accessor_error,
  nd_range_error,
  event_error,
  invalid_parameter_error,
  device_error,
  compile_program_error,
  link_program_error,
  invalid_object_error,
  memory_allocation_error,
  platform_error,
  profiling_error,
  feature_not_supported
};

class source_location {
public:
  source_location(const std::string &function, const std::string &file,
                  int line)
      : _function{function}, _file{file}, _line{line} {}

  const std::string& get_function() const
  { return _function; }

  const std::string& get_file() const
  { return _file; }

  int get_line() const
  { return _line; }

private:
  std::string _function;
  std::string _file;
  int _line;
};

class error_code
{
public:
  error_code(int code)
      : _component{"<unspecified>"}, _has_error_code{true}, _code{code} {}

  error_code() : _component{"<unspecified>"}, _has_error_code{false} {}

  error_code(const std::string &component)
      : _component{component}, _has_error_code{false} {}

  error_code(const std::string &component, int code)
      : _component{component}, _has_error_code{true}, _code{code} {}

  const std::string& get_component() const
  { return _component; }

  bool is_code_specified() const
  { return _has_error_code; }

  int get_code() const
  { return _code; }

  std::string str() const{
    auto res = _component+":";
    if(_has_error_code)
      res += std::to_string(_code);
    else
      res += "<unspecified>";
    return res;
  }

private:
  std::string _component;
  bool _has_error_code;
  int _code;
};

class error_info {
public:
  using errc_type = class error_code;

  error_info() = default;

  error_info(const std::string &message,
            error_type etype = error_type::runtime_error)
      : _message{message}, _etype{etype} {}

  error_info(const std::string &message, errc_type backend_error_code,
             error_type etype = error_type::runtime_error)
      : _message{message}, _error_code{backend_error_code}, _etype{etype} {}

  const std::string& what() const { return _message; }
  errc_type error_code() const { return _error_code; }

  error_type get_error_type() const { return _etype; }
private:
  std::string _message;
  errc_type _error_code;
  error_type _etype;
};

class result {
public:
  // constructs success result
  result() = default;
  result(const source_location &origin, const error_info &info);
  result(const result& other);
  result(result&& other) noexcept;

  friend void swap(result& r1, result& r2) noexcept {
    std::swap(r1._impl, r2._impl);
  }

  result& operator=(const result& other);
  result& operator=(result&& other);

  bool is_success() const;
  
  source_location origin() const;
  error_info info() const;

  std::string what() const;
  void dump(std::ostream& ostr) const;
private:
  struct result_impl
  {
    result_impl(const source_location& l, const error_info& i)
    : origin{l}, info{i}
    {}

    source_location origin;
    error_info info;
  };

  std::unique_ptr<result_impl> _impl;
};

inline result make_success() {
  return result{};
}

inline result make_error(
    const source_location &origin, const error_info &info) {
  return result{origin, info};
}

// Construct an error object and register in the error queue
result register_error(
    const source_location &origin, const error_info &info);
void register_error(const result& err);

inline void print_result(const result& res, bool warn_only = false){

  std::stringstream sstr;
  res.dump(sstr);
  
  if(!res.is_success()) {
    if(!warn_only) { 
      HIPSYCL_DEBUG_ERROR << sstr.str() << std::endl;
    } else {
      HIPSYCL_DEBUG_WARNING << sstr.str() << std::endl;
    }
  } else {
    HIPSYCL_DEBUG_INFO << sstr.str() << std::endl;
  }
}

inline result print_error(const source_location &origin,
                          const error_info &info) {
  result r = make_error(origin, info);
  print_result(r);
  return r;
}

inline result print_warning(const source_location &origin,
                            const error_info &info) {
  result r = make_error(origin, info);
  print_result(r, true);
  return r;
}

}
}

#define __hipsycl_here() ::hipsycl::rt::source_location{__func__, __FILE__, __LINE__}

#endif