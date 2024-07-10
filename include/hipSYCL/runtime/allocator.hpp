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
