/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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


#ifndef HIPSYCL_OCL_HARDWARE_MANAGER_HPP
#define HIPSYCL_OCL_HARDWARE_MANAGER_HPP

#include <vector>
#include <memory>

#include <CL/opencl.hpp>

#include "../hardware.hpp"
#include "ocl_allocator.hpp"
#include "ocl_usm.hpp"

namespace hipsycl {
namespace rt {

class ocl_allocator;
class ocl_hardware_manager;

class ocl_hardware_context : public hardware_context
{
public:
  ocl_hardware_context() = default;
  ocl_hardware_context(const cl::Device &dev, const cl::Context &ctx,
                       int dev_id, int platform_id);
  ocl_hardware_context(const ocl_hardware_context&) = default;

  virtual bool is_cpu() const override;
  virtual bool is_gpu() const override;

  /// \return The maximum number of kernels that can be executed concurrently
  virtual std::size_t get_max_kernel_concurrency() const override;
  /// \return The maximum number of memory transfers that can be executed
  /// concurrently
  virtual std::size_t get_max_memcpy_concurrency() const override;

  virtual std::string get_device_name() const override;
  virtual std::string get_vendor_name() const override;
  virtual std::string get_device_arch() const override;

  virtual bool has(device_support_aspect aspect) const override;
  virtual std::size_t get_property(device_uint_property prop) const override;
  virtual std::vector<std::size_t>
  get_property(device_uint_list_property prop) const override;
  
  virtual std::string get_driver_version() const override;
  virtual std::string get_profile() const override;

  virtual ~ocl_hardware_context();

  ocl_allocator* get_allocator();
  ocl_usm* get_usm_provider();

  int get_platform_id() const;
  int get_device_id() const;
  cl::Device get_cl_device() const;
  cl::Context get_cl_context() const;

  void init_allocator(ocl_hardware_manager* mgr);
private:
  int _dev_id;
  int _platform_id;
  cl::Context _ctx;
  cl::Device _dev;
  std::shared_ptr<ocl_usm> _usm_provider;
  ocl_allocator _alloc;
};

class ocl_hardware_manager : public backend_hardware_manager
{
public:
  ocl_hardware_manager();

  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;

  virtual ~ocl_hardware_manager() {}
  
  cl::Platform get_platform(int platform_id);
  cl::Context get_context(device_id dev);
private:
  std::vector<ocl_hardware_context> _devices;
  std::vector<cl::Platform> _platforms;
  
  hardware_platform _hw_platform;
};

}
}

#endif
