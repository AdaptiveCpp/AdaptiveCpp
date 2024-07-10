/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
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