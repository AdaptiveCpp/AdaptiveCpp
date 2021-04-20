/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef HIPSYCL_INFO_DEVICE_HPP
#define HIPSYCL_INFO_DEVICE_HPP

#include <cstddef>

#include "hipSYCL/sycl/aspect.hpp"
#include "param_traits.hpp"
#include "../types.hpp"
#include "../aspect.hpp"

namespace hipsycl {
namespace sycl {

template<int>
struct id;

class platform;
class device;

namespace info {

enum class device : int {
  device_type,
  vendor_id,
  max_compute_units,
  max_work_item_dimensions,
  max_work_item_sizes,
  max_work_group_size,
  max_num_sub_groups,
  sub_group_independent_forward_progress,
  sub_group_sizes,
  preferred_vector_width_char,
  preferred_vector_width_short,
  preferred_vector_width_int,
  preferred_vector_width_long,
  preferred_vector_width_float,
  preferred_vector_width_double,
  preferred_vector_width_half,
  native_vector_width_char,
  native_vector_width_short,
  native_vector_width_int,
  native_vector_width_long,
  native_vector_width_float,
  native_vector_width_double,
  native_vector_width_half,
  max_clock_frequency,
  address_bits,
  max_mem_alloc_size,
  image_support,
  max_read_image_args,
  max_write_image_args,
  image2d_max_height,
  image2d_max_width,
  image3d_max_height,
  image3d_max_width,
  image3d_max_depth,
  image_max_buffer_size,
  image_max_array_size,
  max_samplers,
  max_parameter_size,
  mem_base_addr_align,
  half_fp_config,
  single_fp_config,
  double_fp_config,
  global_mem_cache_type,
  global_mem_cache_line_size,
  global_mem_cache_size,
  global_mem_size,
  max_constant_buffer_size,
  max_constant_args,
  local_mem_type,
  local_mem_size,
  error_correction_support,
  host_unified_memory,
  profiling_timer_resolution,
  is_endian_little,
  is_available,
  is_compiler_available,
  is_linker_available,
  execution_capabilities,
  queue_profiling,
  built_in_kernels,
  platform,
  name,
  vendor,
  driver_version,
  profile,
  version,
  opencl_c_version,
  aspects,
  extensions,
  printf_buffer_size,
  preferred_interop_user_sync,
  parent_device,
  partition_max_sub_devices,
  partition_properties,
  partition_affinity_domains,
  partition_type_property,
  partition_type_affinity_domain,
  reference_count
};

enum class device_type : unsigned int {
  cpu,
  gpu,
  accelerator,
  custom,
  automatic,
  host,
  all
};

enum class partition_property : int {
  no_partition,
  partition_equally,
  partition_by_counts,
  partition_by_affinity_domain
};

enum class partition_affinity_domain : int {
  not_applicable,
  numa,
  L4_cache,
  L3_cache,
  L2_cache,
  L1_cache,
  next_partitionable
};

enum class local_mem_type : int { none, local, global };

enum class fp_config : int {
  denorm,
  inf_nan,
  round_to_nearest,
  round_to_zero,
  round_to_inf,
  fma,
  correctly_rounded_divide_sqrt,
  soft_float
};

enum class global_mem_cache_type : int { none, read_only, write_only };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::device_type, device_type);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::vendor_id, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_compute_units, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_work_item_dimensions, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_work_item_sizes, id<3>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_work_group_size, size_t);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_num_sub_groups, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::sub_group_independent_forward_progress, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::sub_group_sizes, std::vector<size_t>);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_char, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_double, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_float, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_half, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_int, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_long, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_vector_width_short, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_char, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_double, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_float, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_half, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_int, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_long, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::native_vector_width_short, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_clock_frequency, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::address_bits, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_mem_alloc_size, detail::u_long);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image_support, bool);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_read_image_args, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_write_image_args, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image2d_max_width, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image2d_max_height, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image3d_max_width, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image3d_max_height, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image3d_max_depth, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image_max_buffer_size, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::image_max_array_size, size_t);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_samplers, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_parameter_size, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::mem_base_addr_align, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::half_fp_config, std::vector<fp_config>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::single_fp_config, std::vector<fp_config>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::double_fp_config, std::vector<fp_config>);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::global_mem_cache_type, global_mem_cache_type);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::global_mem_cache_line_size, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::global_mem_cache_size, detail::u_long);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::global_mem_size, detail::u_long);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_constant_buffer_size, detail::u_long);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::max_constant_args, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::local_mem_type, local_mem_type);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::local_mem_size, detail::u_long);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::error_correction_support, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::host_unified_memory, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::profiling_timer_resolution, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::is_endian_little, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::is_available, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::is_compiler_available, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::is_linker_available, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::execution_capabilities, std::vector<execution_capability>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::queue_profiling, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::built_in_kernels, std::vector<string_class>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::platform, sycl::platform);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::name, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::vendor, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::driver_version, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::profile, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::version, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::opencl_c_version, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::aspects, std::vector<aspect>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::extensions, std::vector<string_class>);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::printf_buffer_size, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::preferred_interop_user_sync, bool);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::parent_device, sycl::device);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::partition_max_sub_devices, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::partition_properties, std::vector<partition_property>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::partition_affinity_domains, std::vector<partition_affinity_domain>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::partition_type_property, partition_property);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::partition_type_affinity_domain, partition_affinity_domain);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(device, device::reference_count, detail::u_int);

} // namespace info
} // namespace sycl
} // namespace hipsycl

#endif
