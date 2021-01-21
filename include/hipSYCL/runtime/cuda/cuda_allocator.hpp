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