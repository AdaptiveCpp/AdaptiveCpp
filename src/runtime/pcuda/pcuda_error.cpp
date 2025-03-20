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


#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/async_errors.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl::rt::pcuda {

result make_pcuda_error(const hipsycl::rt::source_location &location,
                          pcudaError_t error, const std::string &message) {
  return make_error(location, error_info("[PCUDA] " + message, error,
                                         error_type::pcuda_error));
}

result make_pcuda_error(const result& internal_err,
                          pcudaError_t error) {
  if(internal_err.is_success())
    return make_success();
  return make_pcuda_error(internal_err.origin(), error,
                          "[PCUDA] " + internal_err.what());
}

void register_pcuda_error(const hipsycl::rt::source_location &location,
                          pcudaError_t error, const std::string &message) {
  rt::register_error(make_pcuda_error(location, error, message));
}

void register_pcuda_error(const result& error, pcudaError_t err) {
  rt::register_error(make_pcuda_error(error, err));
}

pcudaError_t get_most_recent_pcuda_error() {
  pcudaError_t pcuda_err = pcudaSuccess;
  rt::application::errors().for_each_error([&](const rt::result &err) {
    if(err.info().get_error_type() == error_type::pcuda_error)
      pcuda_err = static_cast<pcudaError_t>(err.info().error_code().get_code());
  });
  return pcuda_err;
}

pcudaError_t pop_most_recent_pcuda_error() {
  pcudaError_t pcuda_err = pcudaSuccess;
  rt::application::errors().pop_each_error([&](const rt::result &err) {
    if(err.info().get_error_type() == error_type::pcuda_error)
      pcuda_err = static_cast<pcudaError_t>(err.info().error_code().get_code());
  });
  return pcuda_err;
}

}

