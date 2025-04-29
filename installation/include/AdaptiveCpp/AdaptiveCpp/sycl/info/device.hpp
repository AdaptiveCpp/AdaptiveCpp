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
#ifndef HIPSYCL_INFO_DEVICE_HPP
#define HIPSYCL_INFO_DEVICE_HPP

#include <cstddef>

#include "hipSYCL/sycl/aspect.hpp"
#include "info.hpp"
#include "../types.hpp"
#include "../aspect.hpp"

namespace hipsycl {
namespace sycl {

template<int>
class range;

class platform;
class device;

namespace info {

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

enum class global_mem_cache_type : int { none, read_only, read_write };

enum class execution_capability : unsigned int {
  exec_kernel,
  exec_native_kernel
};

namespace device {
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(device_type, info::device_type);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(vendor_id, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_compute_units, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_work_item_dimensions, detail::u_int);

  template<int Dimensions = 3>
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_work_item_sizes, range<Dimensions>);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_work_group_size, size_t);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_num_sub_groups, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(sub_group_independent_forward_progress, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(sub_group_sizes, std::vector<size_t>);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_char, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_double, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_float, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_half, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_int, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_long, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_vector_width_short, detail::u_int);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_char, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_double, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_float, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_half, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_int, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_long, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(native_vector_width_short, detail::u_int);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_clock_frequency, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(address_bits, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_mem_alloc_size, detail::u_long);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image_support, bool);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_read_image_args, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_write_image_args, detail::u_int);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image2d_max_width, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image2d_max_height, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image3d_max_width, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image3d_max_height, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image3d_max_depth, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image_max_buffer_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(image_max_array_size, size_t);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_samplers, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_parameter_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(mem_base_addr_align, detail::u_int);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(half_fp_config, std::vector<fp_config>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(single_fp_config, std::vector<fp_config>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(double_fp_config, std::vector<fp_config>);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_mem_cache_type, info::global_mem_cache_type);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_mem_cache_line_size, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_mem_cache_size, detail::u_long);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_mem_size, detail::u_long);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_constant_buffer_size, detail::u_long);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_constant_args, detail::u_int);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(local_mem_type, info::local_mem_type);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(local_mem_size, detail::u_long);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(error_correction_support, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(host_unified_memory, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(profiling_timer_resolution, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(is_endian_little, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(is_available, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(is_compiler_available, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(is_linker_available, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(execution_capabilities, std::vector<execution_capability>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(queue_profiling, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(built_in_kernels, std::vector<string_class>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(platform, sycl::platform);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(name, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(vendor, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(driver_version, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(profile, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(version, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(opencl_c_version, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(aspects, std::vector<aspect>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(extensions, std::vector<string_class>);

  HIPSYCL_DEFINE_INFO_DESCRIPTOR(printf_buffer_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_interop_user_sync, bool);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(parent_device, sycl::device);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(partition_max_sub_devices, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(partition_properties, std::vector<partition_property>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(partition_affinity_domains, std::vector<partition_affinity_domain>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(partition_type_property, partition_property);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(partition_type_affinity_domain, partition_affinity_domain);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(reference_count, detail::u_int);
};

} // namespace info
} // namespace sycl
} // namespace hipsycl

#endif
