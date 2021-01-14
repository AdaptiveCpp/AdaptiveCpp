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

#ifndef HIPSYCL_RUNTIME_ALLOCATOR_HPP
#define HIPSYCL_RUNTIME_ALLOCATOR_HPP

#include "device_id.hpp"
#include "error.hpp"

namespace hipsycl {
namespace rt {

struct pointer_info {
  // Device on which memory is allocator. For shared allocations
  // this might not have reliable semantics
  rt::device_id dev;

  bool is_optimized_host;
  bool is_usm;
  // Whether the allocation came from a host backend where
  // semantics can be slightly different (e.g. everything is
  // automatically "usm")
  bool is_from_host_backend;
};

class backend_allocator
{
public:
  virtual void *allocate(size_t min_alignment, size_t size_bytes) = 0;
  // Optimized host memory - may be page-locked, device mapped if supported
  virtual void* allocate_optimized_host(size_t min_alignment, size_t bytes) = 0;
  virtual void free(void *mem) = 0;
  

  /// Allocate memory accessible both from the host and the backend
  virtual void *allocate_usm(size_t bytes) = 0;
  virtual bool is_usm_accessible_from(backend_descriptor b) const = 0;

  // Query the given pointer for its properties. If pointer is unknown,
  // returns non-success result.
  virtual result query_pointer(const void* ptr, pointer_info& out) const = 0;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const = 0;

  virtual ~backend_allocator(){}
};

}
}

#endif
