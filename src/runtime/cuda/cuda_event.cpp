/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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

#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <cuda_runtime_api.h>

namespace hipsycl {
namespace rt {

void cuda_event_deleter::operator()(CUevent_st *evt) const {
  auto err = cudaEventDestroy(evt);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: Couldn't destroy event",
                              error_code{"CUDA", err}});
  }
}

cuda_unique_event make_cuda_event() {
  cudaEvent_t evt;
  if (cudaError_t err = cudaEventCreate(&evt); err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_event: Couldn't create event", error_code{"CUDA", err}});
    return nullptr;
  } else {
    return cuda_unique_event{evt};
  }
}


cuda_node_event::cuda_node_event(device_id dev, cuda_unique_event evt)
: _dev{dev}, _evt{std::move(evt)}
{}

bool cuda_node_event::is_complete() const
{
  cudaError_t err = cudaEventQuery(_evt.get());
  if (err != cudaErrorNotReady && err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: Couldn't query event status",
                              error_code{"CUDA", err}});
  }
  return err == cudaSuccess;
}

void cuda_node_event::wait()
{
  auto err = cudaEventSynchronize(_evt.get());
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_node_event: cudaEventSynchronize() failed",
                              error_code{"CUDA", err}});
  }
}

CUevent_st* cuda_node_event::get_event() const
{
  return _evt.get();
}

device_id cuda_node_event::get_device() const
{
  return _dev;
}

}
}
