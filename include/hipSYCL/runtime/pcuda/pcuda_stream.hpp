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

class stream {
public:
  static pcudaError_t create(stream *&out, pcuda_runtime *, rt::device_id dev,
                             unsigned int flags, int priority);
  static pcudaError_t destroy(stream* s, pcuda_runtime*);
  static pcudaError_t wait_all(device_id dev);
  inorder_queue* get_queue() const;
  static inorder_queue* get_queue(pcudaStream_t stream);
private:
  std::shared_ptr<inorder_executor> _executor;
};

}

#endif
