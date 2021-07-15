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

#ifndef HIPSYCL_HARDWARE_HPP
#define HIPSYCL_HARDWARE_HPP

#include <string>

#include "device_id.hpp"

namespace hipsycl {
namespace rt {

enum class device_support_aspect {
  images,
  error_correction,
  host_unified_memory,
  little_endian,
  global_mem_cache,
  global_mem_cache_read_only,
  global_mem_cache_write_only,
  emulated_local_memory,
  sub_group_independent_forward_progress,
  usm_device_allocations,
  usm_host_allocations,
  usm_atomic_host_allocations,
  usm_shared_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations,
  execution_timestamps
};

enum class device_uint_property {
  max_compute_units,
  max_global_size0,
  max_global_size1,
  max_global_size2,
  max_group_size,
  max_num_sub_groups,
  preferred_vector_width_char,
  preferred_vector_width_double,
  preferred_vector_width_float,
  preferred_vector_width_half,
  preferred_vector_width_int,
  preferred_vector_width_long,
  preferred_vector_width_short,
  native_vector_width_char,
  native_vector_width_double,
  native_vector_width_float,
  native_vector_width_half,
  native_vector_width_int,
  native_vector_width_long,
  native_vector_width_short,
  max_clock_speed,
  max_malloc_size,
  address_bits,

  max_read_image_args,
  max_write_image_args,
  image2d_max_width,
  image2d_max_height,
  image3d_max_width,
  image3d_max_height,
  image3d_max_depth,
  image_max_buffer_size,
  image_max_array_size,
  max_samplers,

  max_parameter_size,
  mem_base_addr_align,
  global_mem_cache_line_size,
  global_mem_cache_size,
  global_mem_size,

  max_constant_buffer_size,
  max_constant_args,

  local_mem_size,

  printf_buffer_size,
  partition_max_sub_devices,

  vendor_id
};

enum class device_uint_list_property {
  sub_group_sizes
};

class hardware_context
{
public:
  virtual bool is_cpu() const = 0;
  virtual bool is_gpu() const = 0;

  /// \return The maximum number of kernels that can be executed concurrently
  virtual std::size_t get_max_kernel_concurrency() const = 0;
  /// \return The maximum number of memory transfers that can be executed
  /// concurrently
  virtual std::size_t get_max_memcpy_concurrency() const = 0;

  virtual std::string get_device_name() const = 0;
  virtual std::string get_vendor_name() const = 0;
  virtual std::string get_device_arch() const = 0;

  virtual bool has(device_support_aspect aspect) const = 0;
  
  virtual std::size_t get_property(device_uint_property prop) const = 0;

  virtual std::vector<std::size_t>
  get_property(device_uint_list_property prop) const = 0;

  virtual std::string get_driver_version() const = 0;
  virtual std::string get_profile() const = 0;
  
  virtual ~hardware_context(){}
};

class backend_hardware_manager
{
public:
  virtual std::size_t get_num_devices() const = 0;
  virtual hardware_context *get_device(std::size_t index) = 0;
  virtual device_id get_device_id(std::size_t index) const = 0;

  virtual ~backend_hardware_manager(){}
};



}
}

#endif
