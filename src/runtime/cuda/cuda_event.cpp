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
#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/cuda/cuda_event_pool.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cuda_runtime_api.h>

namespace hipsycl {
namespace rt {


cuda_node_event::cuda_node_event(device_id dev, cudaEvent_t evt, cuda_event_pool* pool)
: _dev{dev}, _evt{evt}, _pool{pool}
{}

cuda_node_event::~cuda_node_event() {
  if(_pool) {
    _pool->release_event(_evt);
  }
}

bool cuda_node_event::is_complete() const
{
  cudaError_t err = cudaEventQuery(_evt);
  if (err != cudaErrorNotReady && err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_node_event: Couldn't query event status",
                              error_code{"CUDA", err}});
  }
  return err == cudaSuccess;
}

void cuda_node_event::wait()
{
  auto err = cudaEventSynchronize(_evt);
  if (err != cudaSuccess) {
    register_error(__acpp_here(),
                   error_info{"cuda_node_event: cudaEventSynchronize() failed",
                              error_code{"CUDA", err}});
  }
}

cuda_node_event::backend_event_type cuda_node_event::get_event() const
{
  return _evt;
}

device_id cuda_node_event::get_device() const
{
  return _dev;
}

cuda_node_event::backend_event_type
cuda_node_event::request_backend_event() {
  return get_event();
}

}
}
