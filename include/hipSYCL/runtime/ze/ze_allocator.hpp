/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_ZE_ALLOCATOR_HPP
#define HIPSYCL_ZE_ALLOCATOR_HPP

#include "../allocator.hpp"
#include "ze_hardware_manager.hpp"

#include <level_zero/ze_api.h>

namespace hipsycl {
namespace rt {

class ze_allocator : public backend_allocator 
{
public:
  ze_allocator(const ze_hardware_context* dev, const ze_hardware_manager* hw_manager);

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
  ze_context_handle_t _ctx;
  ze_device_handle_t _dev;
  uint32_t _global_mem_ordinal;

  const ze_hardware_manager* _hw_manager;
};

}
}

#endif