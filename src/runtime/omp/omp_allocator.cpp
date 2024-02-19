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
#include <cstdlib>

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/omp/omp_allocator.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

omp_allocator::omp_allocator(const device_id &my_device)
    : _my_device{my_device} {}

void *omp_allocator::allocate(size_t min_alignment, size_t size_bytes) {
#if !defined(_WIN32)
  // posix requires alignment to be a multiple of sizeof(void*)
  if (min_alignment < sizeof(void*))
    return std::malloc(size_bytes);
#else
  /* The std::free function of the Microsoft C Runtime Library cannot handle
     aligned memory, therefore omp_allocator::free always calls _aligned_free.
     This, however, can only free memory allocated with _aligned_malloc, but
     _aligned_malloc returns NULL when min_alignment == 0.  */
  if (min_alignment == 0)
    min_alignment = 1;
#endif

  if(size_bytes % min_alignment != 0)
    return nullptr;

  // ToDo: Mac OS CI has a problem with std::aligned_alloc
  // but it's unclear if it's a Mac, or libc++, or toolchain issue
#ifdef __APPLE__
  return aligned_alloc(min_alignment, size_bytes);
#elif !defined(_WIN32)
  return std::aligned_alloc(min_alignment, size_bytes);
#else
  min_alignment = power_of_2_ceil(min_alignment);
  return _aligned_malloc(size_bytes, min_alignment);
#endif
}

void *omp_allocator::allocate_optimized_host(size_t min_alignment,
                                             size_t bytes) {
  return this->allocate(min_alignment, bytes);
};

void omp_allocator::free(void *mem) {
#if !defined(_WIN32)
  std::free(mem);
#else
  _aligned_free(mem);
#endif
}

void* omp_allocator::allocate_usm(size_t bytes) {
  return this->allocate(0, bytes);
}

bool omp_allocator::is_usm_accessible_from(backend_descriptor b) const {
  if(b.hw_platform == hardware_platform::cpu) {
    return true;
  }
  return false;
}

result omp_allocator::query_pointer(const void *ptr, pointer_info &out) const {
  
  // For a host device, USM is the same as host memory?
  out.is_optimized_host = true;
  out.is_usm = true;
  out.is_from_host_backend = true;
  out.dev = _my_device;

  return make_success();
}

result omp_allocator::mem_advise(const void *addr, std::size_t num_bytes,
                                 int advise) const {
  HIPSYCL_DEBUG_WARNING << "omp_allocator: Ignoring mem_advise() hint"
                        << std::endl;
  return make_success();
}

}
}
