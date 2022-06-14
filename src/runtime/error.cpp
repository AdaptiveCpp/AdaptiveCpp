/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/async_errors.hpp"

namespace hipsycl {
namespace rt {

result::result(const source_location &origin, const error_info &info)
: _impl{std::make_unique<result_impl>(origin, info)}
{}

result::result(const result& other){
  if(other._impl)
    _impl =
        std::make_unique<result_impl>(other._impl->origin, other._impl->info);
}

result::result(result&& other) noexcept
: _impl{std::move(other._impl)}
{}

result &result::operator=(const result &other) {
  result r = other;
  swap(*this, r);
  return *this;
}

result& result::operator=(result&& other)
{
  swap(*this, other );
  return *this;
}

bool result::is_success() const {
  return _impl == nullptr;
}

source_location result::origin() const {
  if(!_impl)
    return source_location{"<unspecified>", "<unspecified>", -1};
  return _impl->origin;
}

error_info result::info() const {
  if(!_impl)
    return error_info{};
  return _impl->info;
}

std::string result::what() const {
  std::stringstream sstream;
  this->dump(sstream);
  return sstream.str();
}

void result::dump(std::ostream& ostr) const {
  if(is_success()) ostr << "[success] ";
  else {
    ostr << "from " << _impl->origin.get_file() << ":"
          << _impl->origin.get_line() << " @ " << _impl->origin.get_function()
          << "(): " << _impl->info.what();
    if (_impl->info.error_code().is_code_specified())
      ostr << " (error code = " << _impl->info.error_code().str() << ")";
  }
}

result register_error(
    const source_location &origin, const error_info &info) {

  auto res = make_error(origin, info);

  application::errors().add(res);
  return res;
}

void register_error(const result &err) {
  application::errors().add(err);
}

}
}