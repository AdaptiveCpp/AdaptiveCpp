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

#include <musa_runtime_api.h>

#include "hipSYCL/runtime/musa/musa_allocator.hpp"
#include "hipSYCL/runtime/musa/musa_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

musa_allocator::musa_allocator(backend_descriptor desc, int musa_device)
    : _backend_descriptor{desc}, _dev{musa_device}
{}
      
void *musa_allocator::allocate(size_t min_alignment, size_t size_bytes)
{
  void *ptr;
  musa_device_manager::get().activate_device(_dev);
  musaError_t err = musaMalloc(&ptr, size_bytes);

  if (err != musaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"musa_allocator: musaMalloc() failed",
                              error_code{"MUSA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

void *musa_allocator::allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  void *ptr;
  musa_device_manager::get().activate_device(_dev);

  musaError_t err = musaMallocHost(&ptr, bytes);

  if (err != musaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"musa_allocator: musaMallocHost() failed",
                              error_code{"MUSA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void musa_allocator::free(void *mem) {

  pointer_info info;
  result query_result = query_pointer(mem, info);

  if (!query_result.is_success()) {
    register_error(query_result);
    return;
  }
  
  musaError_t err;
  if (info.is_optimized_host)
    err = musaFreeHost(mem);
  else
    err = musaFree(mem);
  
  if (err != musaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"musa_allocator: musaFree() failed",
                              error_code{"MUSA", err},
                              error_type::memory_allocation_error});
  }
}

void * musa_allocator::allocate_usm(size_t bytes)
{
  musa_device_manager::get().activate_device(_dev);
  
  void *ptr;
  auto err = musaMallocManaged(&ptr, bytes);
  if (err != musaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"musa_allocator: musaMallocManaged() failed",
                              error_code{"MUSA", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

bool musa_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

result musa_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  musaPointerAttributes attribs;
  auto err = musaPointerGetAttributes(&attribs, ptr);

  if (err != musaSuccess) {
    if (err == musaErrorInvalidValue)
      return make_error(
          __hipsycl_here(),
          error_info{"musa_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"MUSA", err},
                     error_type::invalid_parameter_error});
    else
      return make_error(
          __hipsycl_here(),
          error_info{"musa_allocator: query_pointer(): query failed",
                     error_code{"MUSA", err}});
  }

  // CUDA 11+ return cudaMemoryTypeUnregistered and cudaSuccess
  // for unknown host pointers
  if (attribs.type == musaMemoryTypeUnregistered) {
    return make_error(
          __hipsycl_here(),
          error_info{"musa_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"MUSA", err},
                     error_type::invalid_parameter_error});
  }

  out.dev = rt::device_id{_backend_descriptor, attribs.device};
  out.is_from_host_backend = false;
  out.is_optimized_host = attribs.type == musaMemoryTypeHost;
  out.is_usm = attribs.type == musaMemoryTypeManaged;
  
  return make_success();
}

result musa_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const {
#ifndef _WIN32
  musaError_t err = musaMemAdvise(addr, num_bytes,
                                  static_cast<musaMemoryAdvise>(advise), _dev);
  if(err != musaSuccess) {
    return make_error(
      __hipsycl_here(),
      error_info{"musa_allocator: musaMemAdvise() failed", error_code{"MUSA", err}}
    );
  }
#else
  HIPSYCL_DEBUG_WARNING << "musa_allocator: Ignoring mem_advise() hint"
                        << std::endl;
#endif // _WIN32
  return make_success();
}

}
}
