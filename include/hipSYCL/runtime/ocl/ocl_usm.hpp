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

#ifndef HIPSYCL_OCL_USM_HPP
#define HIPSYCL_OCL_USM_HPP

#include <CL/opencl.hpp>
#include "../allocator.hpp"
#include <memory>

namespace hipsycl {
namespace rt {

struct ocl_hardware_manager;

class ocl_usm {
public:
  virtual ~ocl_usm(){}
  virtual bool is_available() const = 0;
  virtual bool has_usm_device_allocations() const = 0;
  virtual bool has_usm_host_allocations() const = 0;
  virtual bool has_usm_atomic_host_allocations() const = 0;
  virtual bool has_usm_shared_allocations() const = 0;
  virtual bool has_usm_atomic_shared_allocations() const = 0;
  virtual bool has_usm_system_allocations() const = 0;

  virtual void* malloc_host(std::size_t size, std::size_t alignment, cl_int& err) = 0;
  virtual void* malloc_device(std::size_t size, std::size_t alignment, cl_int& err) = 0;
  virtual void* malloc_shared(std::size_t size, std::size_t alignment, cl_int& err) = 0;
  virtual cl_int free(void* ptr) = 0;

  virtual cl_int get_alloc_info(const void* ptr, pointer_info& out) = 0;

  virtual cl_int enqueue_memcpy(cl::CommandQueue &queue, void *dst,
                              const void *src, std::size_t size,
                              const std::vector<cl::Event>& wait_events,
                              cl::Event *evt_out) = 0;

  virtual cl_int enqueue_memset(cl::CommandQueue &queue, void *ptr,
                              cl_int pattern, std::size_t bytes,
                              const std::vector<cl::Event> &wait_events,
                              cl::Event *out) = 0;

  virtual cl_int enqueue_prefetch(cl::CommandQueue &queue, const void *ptr,
                                  std::size_t bytes,
                                  cl_mem_migration_flags flags,
                                  const std::vector<cl::Event> &wait_events,
                                  cl::Event *out) = 0;

  virtual cl_int enable_indirect_usm_access(cl::Kernel&) = 0;

  static std::unique_ptr<ocl_usm> from_intel_extension(ocl_hardware_manager* hw_mgr, int device_index);
  static std::unique_ptr<ocl_usm> from_fine_grained_system_svm(ocl_hardware_manager* hw_mgr, int device_index);
};


}
}

#endif