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

#ifndef ACPP_RT_PCUDA_STREAM_HPP
#define ACPP_RT_PCUDA_STREAM_HPP

#include <memory>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/inorder_executor.hpp"


namespace hipsycl::rt::pcuda {

class pcuda_runtime;
using internal_stream_t = std::shared_ptr<inorder_executor>;

pcudaError_t stream_wait_all(rt::device_id dev);

pcudaError_t stream_create(internal_stream_t *&out, pcuda_runtime *,
                           rt::device_id dev, unsigned int flags, int priority);
pcudaError_t stream_destroy(internal_stream_t *stream, pcuda_runtime *);

inorder_queue* stream_get(pcudaStream_t stream);

}

#endif
