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