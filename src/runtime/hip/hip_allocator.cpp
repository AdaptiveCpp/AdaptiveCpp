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

#include "hipSYCL/runtime/hip/hip_allocator.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/exception.hpp"

namespace hipsycl {
namespace rt {

hip_allocator::hip_allocator(int hip_device)
    : _dev{hip_device}
{}
      
void *hip_allocator::allocate(size_t min_alignment, size_t size_bytes)
{
  void *ptr;
  hip_device_manager::get().activate_device(_dev);
  sycl::detail::check_error(hipMalloc(&ptr, size_bytes));

  return ptr;
}

void hip_allocator::free(void *mem)
{
  sycl::detail::check_error(hipFree(mem));
}

void * hip_allocator::allocate_usm(size_t bytes)
{
  void *ptr;
  sycl::detail::check_error(hipMallocManaged(&ptr, bytes));

  return ptr;
}

bool hip_allocator::is_usm_accessible_from(backend_descriptor b) const
{
  // TODO: Formulate this better - this assumes that either CUDA+CPU or
  // ROCm + CPU are active at the same time
  return true;
}

}
}
