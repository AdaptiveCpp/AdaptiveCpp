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

namespace hipsycl {
namespace rt {

class result_origin {
public:
  result_origin(const std::string& function, const std::string& file, int line)
  : _function{function}, _file{file}, _line{line}
  {}

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

class result_info {
public:
  result_info()
  : _message{}, _error_code{0} {}

  result_info(const std::string& message, int backend_error_code) 
  : _message{message}, _error_code{backend_error_code} {}

  const std::string& what() const { return _message; }
  int error_code() const { return _error_code; }
private:
  std::string _message;
  int _error_code;
};

class result {
public:
  // constructs success result
  result(const result_origin& origin)
  : _is_success{true}, _origin{origin}, _error_handler{[](const result&){}}
  {}

  template <class Error_handler>
  result(const result_origin &origin, const result_info &info,
         Error_handler handler)
  : _origin{origin}, _info{info}, _error_handler{handler}
  {}

  bool is_success() const {
    return _is_success;
  }

  result_origin origin() const {
    return _origin;
  }

  result_info info() const {
    return _info;
  }

  void invoke_handler() const {
    _error_handler(*this);
  }

  void dump(std::ostream& ostr) const {
    if(_is_success) ostr << "[success]";
    else {
      ostr << "from " << _origin.get_file() << ":" << _origin.get_line()
           << " @ " << _origin.get_function() << ": " << _info.what()
           << " (error code = " << _info.error_code() << ")" << std::endl;
    }
  }
private:
  bool _is_success;
  result_origin _origin;
  result_info _info;
  std::function<void (const result&)> _error_handler;
};

inline result make_success(const result_origin& origin) {
  return result{origin};
}

inline result make_success(){
  return make_success(result_origin{"<unknown-function>", "<unknown-file>", -1});
}

template <class Error_handler>
result make_error(
    const result_origin &origin, const result_info &info,
    Error_handler handler = [](const result &) {}) {
  return result{origin, info, handler};
}


}
}

#define hipsycl_here() ::hipsycl::rt::result_origin{__func__, __FILE__, __LINE__}

#endif