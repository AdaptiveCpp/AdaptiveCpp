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

#include <cuda_runtime_api.h>

#include "hipSYCL/runtime/cuda/cuda_allocator.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

cuda_allocator::cuda_allocator(backend_descriptor desc, int cuda_device)
    : _backend_descriptor{desc}, _dev{cuda_device}
{}
      
void *cuda_allocator::allocate(size_t min_alignment, size_t size_bytes)
{
  void *ptr;
  cuda_device_manager::get().activate_device(_dev);
  cudaError_t err = cudaMalloc(&ptr, size_bytes);

  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_allocator: cudaMalloc() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

void *cuda_allocator::allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  void *ptr;
  cuda_device_manager::get().activate_device(_dev);

  cudaError_t err = cudaMallocHost(&ptr, bytes);

  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_allocator: cudaMallocHost() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void cuda_allocator::free(void *mem) {

  pointer_info info;
  result query_result = query_pointer(mem, info);

  if (!query_result.is_success()) {
    register_error(query_result);
    return;
  }
  
  cudaError_t err;
  if (info.is_optimized_host)
    err = cudaFreeHost(mem);
  else
    err = cudaFree(mem);
  
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_allocator: cudaFree() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
  }
}

void * cuda_allocator::allocate_usm(size_t bytes)
{
  cuda_device_manager::get().activate_device(_dev);
  
  void *ptr;
  auto err = cudaMallocManaged(&ptr, bytes);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_allocator: cudaMallocManaged() failed",
                              error_code{"CUDA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

bool cuda_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

result cuda_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  cudaPointerAttributes attribs;
  auto err = cudaPointerGetAttributes(&attribs, ptr);

  if (err != cudaSuccess) {
    if (err == cudaErrorInvalidValue)
      return make_error(
          __hipsycl_here(),
          error_info{"cuda_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"CUDA", err},
                     error_type::invalid_parameter_error});
    else
      return make_error(
          __hipsycl_here(),
          error_info{"cuda_allocator: query_pointer(): query failed",
                     error_code{"CUDA", err}});
  }

  // CUDA 11+ return cudaMemoryTypeUnregistered and cudaSuccess
  // for unknown host pointers
  if (attribs.type == cudaMemoryTypeUnregistered) {
    return make_error(
          __hipsycl_here(),
          error_info{"cuda_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"CUDA", err},
                     error_type::invalid_parameter_error});
  }

  out.dev = rt::device_id{_backend_descriptor, attribs.device};
  out.is_from_host_backend = false;
  out.is_optimized_host = attribs.type == cudaMemoryTypeHost;
  out.is_usm = attribs.type == cudaMemoryTypeManaged;
  
  return make_success();
}

result cuda_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const {
#ifndef _WIN32
  cudaError_t err = cudaMemAdvise(addr, num_bytes,
                                  static_cast<cudaMemoryAdvise>(advise), _dev);
  if(err != cudaSuccess) {
    return make_error(
      __hipsycl_here(),
      error_info{"cuda_allocator: cudaMemAdvise() failed", error_code{"CUDA", err}}
    );
  }
#else
  HIPSYCL_DEBUG_WARNING << "cuda_allocator: Ignoring mem_advise() hint"
                        << std::endl;
#endif // _WIN32
  return make_success();
}

}
}
