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
#ifndef HIPSYCL_METAL_HARDWARE_MANAGER_HPP
#define HIPSYCL_METAL_HARDWARE_MANAGER_HPP

#include "../hardware.hpp"

#include "metal_queue.hpp"
#include "metal_allocator.hpp"

namespace MTL {

class Device;

} // namespace MTL

namespace hipsycl {
namespace rt {

class metal_hardware_context : public hardware_context
{
public:
  metal_hardware_context(MTL::Device* device);

  virtual bool is_cpu() const override;
  virtual bool is_gpu() const override;

  virtual std::size_t get_max_kernel_concurrency() const override;
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

  virtual std::size_t get_platform_index() const override;

  virtual ~metal_hardware_context();

private:
  MTL::Device* _device = nullptr;
  uint64_t _core_count;
  uint64_t _max_clock_speed;
  uint64_t _slc_size;
  uint64_t _max_allocated_size;
  int _gpu_family;
};

class metal_hardware_manager : public backend_hardware_manager
{
public:
  metal_hardware_manager();
  virtual std::size_t get_num_devices() const override;
  virtual hardware_context *get_device(std::size_t index) override;
  virtual device_id get_device_id(std::size_t index) const override;
  virtual std::size_t get_num_platforms() const override;

  virtual ~metal_hardware_manager();
private:
  friend class metal_backend;
  metal_allocator* get_allocator(size_t index);
  metal_inorder_queue* make_queue(size_t index);

  std::vector<MTL::Device*> _devices;
  std::vector<metal_hardware_context> _contexts;
  std::deque<metal_allocator> _allocators;
};

} // namespace rt
} // namespace hipsycl

#endif
