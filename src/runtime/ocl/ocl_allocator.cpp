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

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"

#include "hipSYCL/runtime/ocl/ocl_allocator.hpp"
#include <cstddef>

namespace hipsycl {
namespace rt {

ocl_allocator::ocl_allocator(ocl_usm* usm)
: _usm{usm} {}

void* ocl_allocator::allocate(size_t min_alignment, size_t size_bytes) {
  if(!_usm->is_available()) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  
  cl_int err;
  void* ptr = _usm->malloc_device(size_bytes, min_alignment, err);
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: USM device allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void *ocl_allocator::allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  if(!_usm->is_available()) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  cl_int err;
  void* ptr = _usm->malloc_host(bytes, min_alignment, err);
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: USM host allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void ocl_allocator::free(void *mem) {
  if(!_usm->is_available()) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return;
  }
  cl_int err = _usm->free(mem);
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: USM memory free() failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
  }
}

void *ocl_allocator::allocate_usm(size_t bytes) {
  if(!_usm->is_available()) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    return nullptr;
  }
  cl_int err;
  void* ptr = _usm->malloc_shared(bytes, 0, err);
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_allocator: USM shared allocation failed",
                              error_code{"CL", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

bool ocl_allocator::is_usm_accessible_from(backend_descriptor b) const {
  // TODO: This is INCORRECT. We would need to compare the OpenCL platform, not
  // just the backend descriptor. The current API does not allow for that.
  // Probably, we should change this function to accept a device_id.
  // Or just remove this function entirely, as it does not seem to be used?
  return b.hw_platform == hardware_platform::ocl;
}

result ocl_allocator::query_pointer(const void* ptr, pointer_info& out) const {
  if(!_usm->is_available()) {
    auto err = make_error(__hipsycl_here(),
                   error_info{"ocl_allocator: OpenCL device does not have valid USM provider",
                              error_type::memory_allocation_error});
    register_error(err);
    return err;
  }
  cl_int err = _usm->get_alloc_info(ptr, out);
  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ocl_allocator: query_pointer(): query failed",
                     error_code{"CL", err}});
  }
  return make_success();
}

result ocl_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                          int advise) const {
  // TODO
  return make_success();
}


}
}

