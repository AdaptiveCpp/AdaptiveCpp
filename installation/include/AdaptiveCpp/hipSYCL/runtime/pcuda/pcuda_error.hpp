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

#ifndef ACPP_RT_PCUDA_ERROR_HPP
#define ACPP_RT_PCUDA_ERROR_HPP

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/error.hpp"
namespace hipsycl::rt::pcuda {

result make_pcuda_error(const hipsycl::rt::source_location &location,
                          pcudaError_t error, const std::string &message);
result make_pcuda_error(const result& internal_err, pcudaError_t error);
// TODO CUDA errors are thread-local, whereas our errors are application-wide
void register_pcuda_error(const hipsycl::rt::source_location &location,
                          pcudaError_t error, const std::string &message);
void register_pcuda_error(const result& error, pcudaError_t err);

pcudaError_t get_most_recent_pcuda_error();
pcudaError_t pop_most_recent_pcuda_error();

}

#endif