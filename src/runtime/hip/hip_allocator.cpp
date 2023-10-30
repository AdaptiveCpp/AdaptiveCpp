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

#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/hip/hip_allocator.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

hip_allocator::hip_allocator(backend_descriptor desc, int hip_device)
    : _backend_descriptor{desc}, _dev{hip_device}
{}
      
void *hip_allocator::allocate(size_t min_alignment, size_t size_bytes)
{
  void *ptr;
  hip_device_manager::get().activate_device(_dev);
  hipError_t err = hipMalloc(&ptr, size_bytes);

  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_allocator: hipMalloc() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

void *hip_allocator::allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  void *ptr;
  hip_device_manager::get().activate_device(_dev);

  hipError_t err = hipHostMalloc(&ptr, bytes, hipHostMallocDefault);

  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_allocator: hipHostMalloc() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }
  return ptr;
}

void hip_allocator::free(void *mem) {

  pointer_info info;
  result query_result = query_pointer(mem, info);

  if (!query_result.is_success()) {
    register_error(query_result);
    return;
  }
  
  hipError_t err;
  if (info.is_optimized_host)
    err = hipHostFree(mem);
  else
    err = hipFree(mem);
  
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_allocator: hipFree() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
  }
}

void * hip_allocator::allocate_usm(size_t bytes)
{
  hip_device_manager::get().activate_device(_dev);

  void *ptr;
  auto err = hipMallocManaged(&ptr, bytes);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_allocator: hipMallocManaged() failed",
                              error_code{"HIP", err},
                              error_type::memory_allocation_error});
    return nullptr;
  }

  return ptr;
}

bool hip_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

result hip_allocator::query_pointer(const void *ptr, pointer_info &out) const
{
  hipPointerAttribute_t attribs;

  auto err = hipPointerGetAttributes(&attribs, ptr);

  if (err != hipSuccess) {
    if (err == hipErrorInvalidValue)
      return make_error(
          __hipsycl_here(),
          error_info{"hip_allocator: query_pointer(): pointer is unknown by backend",
                     error_code{"HIP", err},
                     error_type::invalid_parameter_error});
    else
      return make_error(
          __hipsycl_here(),
          error_info{"hip_allocator: query_pointer(): query failed",
                     error_code{"HIP", err}});
  }
#if HIP_VERSION_MAJOR > 5
  const auto memoryType = attribs.type;
#else
  const auto memoryType = attribs.memoryType;
#endif

  out.dev = rt::device_id{_backend_descriptor, attribs.device};
  out.is_from_host_backend = false;
  out.is_optimized_host = (memoryType == hipMemoryTypeHost);
#ifndef HIPSYCL_RT_NO_HIP_MANAGED_MEMORY
  out.is_usm = attribs.isManaged;
#else
  // query for hipMemoryTypeUnified as dummy; this is not actually
  // correct as HIP versions that do not support attribs.isManaged
  // have no way of querying whether something was malloc'd with
  // hipMallocManaged().
  out.is_usm = (memoryType == hipMemoryTypeUnified);
#endif
  
  return make_success();
}

result hip_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                int advise) const {
#ifndef HIPSYCL_RT_NO_HIP_MANAGED_MEMORY
  hipError_t err = hipMemAdvise(addr, num_bytes,
                                static_cast<hipMemoryAdvise>(advise), _dev);
  if(err != hipSuccess) {
    return make_error(
      __hipsycl_here(),
      error_info{"hip_allocator: hipMemAdvise() failed", error_code{"HIP", err}}
    );
  }
#else
  HIPSYCL_DEBUG_WARNING << "hip_allocator: Ignoring mem_advise() hint"
                        << std::endl;
#endif
  return make_success();
}

}
}
