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


#ifndef HIPSYCL_ZE_HARDWARE_MANAGER_HPP
#define HIPSYCL_ZE_HARDWARE_MANAGER_HPP

#include <vector>
#include <memory>
#include <level_zero/ze_api.h>

#include "../hardware.hpp"
#include "../error.hpp"

namespace hipsycl {
namespace rt {


class ze_context_manager
{
public:
  ze_context_manager(ze_driver_handle_t driver);
  ze_context_handle_t get() const;
  ze_driver_handle_t get_driver() const;

  ~ze_context_manager();
private:
  std::shared_ptr<ze_context_handle_t> _handle;
  ze_driver_handle_t _driver;
};

class ze_event_pool_manager
{
public:
  ze_event_pool_manager(ze_context_handle_t ctx,
                        const std::vector<ze_device_handle_t>& _devices,
                        std::size_t pool_size = 128);
  ~ze_event_pool_manager();

  std::shared_ptr<ze_event_pool_handle_t> get_pool() const;

  ze_context_handle_t get_ze_context() const;

  std::shared_ptr<ze_event_pool_handle_t>
  allocate_event(uint32_t &event_ordinal);

private:
  void spawn_pool();

  std::vector<ze_device_handle_t> _devices;
  ze_context_handle_t _ctx;
  std::shared_ptr<ze_event_pool_handle_t> _pool;

  std::size_t _pool_size;
  uint32_t _num_used_events;
};

class ze_hardware_context : public hardware_context
{
public:
  ze_hardware_context(ze_driver_handle_t driver, ze_device_handle_t device,
                      ze_context_handle_t ctx);

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

  virtual ~ze_hardware_context();

  ze_driver_handle_t get_ze_driver() const
  { return _driver; }

  ze_device_handle_t get_ze_device() const
  { return _device; }

  ze_context_handle_t get_ze_context() const
  { return _ctx; }

  uint32_t get_ze_global_memory_ordinal() const;

private:
  ze_driver_handle_t _driver;
  ze_device_handle_t _device;
  ze_context_handle_t _ctx;

  ze_device_properties_t _props;
  ze_device_compute_properties_t _compute_props;
  std::vector<ze_device_memory_properties_t> _memory_props;
};

class ze_hardware_manager : public backend_hardware_manager
{
public:
  ze_hardware_manager();

  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;

  virtual ~ze_hardware_manager() {}
  
  ze_context_handle_t get_ze_context(std::size_t device_index) const;
  result device_handle_to_device_id(ze_device_handle_t d, device_id &out) const;
  ze_event_pool_manager* get_event_pool_manager(std::size_t device_index);

private:
  std::vector<ze_hardware_context> _devices;
  std::vector<ze_driver_handle_t> _drivers;
  std::vector<ze_context_manager> _contexts;
  std::vector<ze_event_pool_manager> _event_pools;

  hardware_platform _hw_platform;
};

}
}

#endif
