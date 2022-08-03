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


#ifndef HIPSYCL_HIP_HARDWARE_MANAGER_HPP
#define HIPSYCL_HIP_HARDWARE_MANAGER_HPP

#include <vector>
#include <memory>

#include "../hardware.hpp"
#include "hip_target.hpp"

struct hipDeviceProp_t;

namespace hipsycl {
namespace rt {

class hip_allocator;
class hip_event_pool;

class hip_hardware_context : public hardware_context
{
public:
  hip_hardware_context() = default;
  hip_hardware_context(int dev);
  hip_hardware_context(hip_hardware_context&&) = default;

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

  virtual ~hip_hardware_context() {}

  hip_allocator* get_allocator() const;
  hip_event_pool* get_event_pool() const;
private:
  std::unique_ptr<hipDeviceProp_t> _properties;
  std::unique_ptr<hip_allocator> _allocator;
  std::unique_ptr<hip_event_pool> _event_pool;
  int _dev;
};

class hip_hardware_manager : public backend_hardware_manager
{
public:
  hip_hardware_manager(hardware_platform hw_platform);

  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;

  virtual ~hip_hardware_manager() {}
  
private:
  std::vector<hip_hardware_context> _devices;
  hardware_platform _hw_platform;
};

}
}

#endif
