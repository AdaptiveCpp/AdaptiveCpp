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
#ifndef HIPSYCL_CUDA_ALLOCATOR_HPP
#define HIPSYCL_CUDA_ALLOCATOR_HPP

#include "../allocator.hpp"

namespace hipsycl {
namespace rt {

class cuda_allocator : public backend_allocator 
{
public:
  cuda_allocator(backend_descriptor desc, int cuda_device);

  virtual void* allocate(size_t min_alignment, size_t size_bytes) override;

  virtual void *allocate_optimized_host(size_t min_alignment,
                                        size_t bytes) override;
  
  virtual void free(void *mem) override;

  virtual void *allocate_usm(size_t bytes) override;
  virtual bool is_usm_accessible_from(backend_descriptor b) const override;

  virtual result query_pointer(const void* ptr, pointer_info& out) const override;

  virtual result mem_advise(const void *addr, std::size_t num_bytes,
                            int advise) const override;
private:
  backend_descriptor _backend_descriptor;
  int _dev;
};

}
}

#endif