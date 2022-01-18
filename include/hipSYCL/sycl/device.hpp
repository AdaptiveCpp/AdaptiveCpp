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


#ifndef HIPSYCL_DEVICE_HPP
#define HIPSYCL_DEVICE_HPP

#include <exception>
#include <limits>
#include <type_traits>

#include "types.hpp"
#include "aspect.hpp"
#include "info/info.hpp"
#include "backend.hpp"
#include "exception.hpp"
#include "version.hpp"
#include "hipSYCL/sycl/libkernel/id.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/hardware.hpp"

namespace hipsycl {
namespace sycl {

class device;

namespace detail {

inline rt::device_id get_host_device() {
  return rt::device_id{rt::backend_descriptor(rt::hardware_platform::cpu,
                                              rt::api_platform::omp),
                       0};
}

rt::device_id extract_rt_device(const device&);

}

class device_selector;
class platform;

class device {
  friend class queue;
  friend class context;
  friend class platform;
  friend rt::device_id detail::extract_rt_device(const device&);
public:
  device(rt::device_id id)
      : _device_id{id} {}
 
  device()
      : _device_id(detail::get_host_device()) {}

  // Implemented in device_selector.hpp
  template <class DeviceSelector>
  explicit device(const DeviceSelector &deviceSelector);

  bool is_host() const 
  {
    return is_cpu();
  }

  bool is_cpu() const
  {
    return get_rt_device()->is_cpu();
  }

  bool is_gpu() const
  {
    return get_rt_device()->is_gpu();
  }

  bool is_accelerator() const { return !is_cpu(); }

  bool has(aspect asp) const {
    if(asp == aspect::cpu) {
      return is_cpu();
    } else if(asp == aspect::gpu) {
      return is_gpu();
    } else if(asp == aspect::accelerator) {
      return is_accelerator();
    } else if(asp == aspect::custom) {
      return false;
    } else if(asp == aspect::emulated) {
      return false;
    } else if(asp == aspect::host_debuggable) {
      return _device_id.get_full_backend_descriptor().hw_platform ==
           rt::hardware_platform::cpu;
    } else if(asp == aspect::fp16) {
      // fp16 is only partially supported in hipSYCL
      return false;
    } else if(asp == aspect::fp64) {
      return true;
    } else if(asp == aspect::atomic64) {
      return true;
    } else if(asp == aspect::image) {
      return false;
    } else if(asp == aspect::online_compiler) {
      return false;
    } else if(asp == aspect::online_linker) {
      return false;
    } else if(asp == aspect::queue_profiling) {
      return get_rt_device()->has(
          rt::device_support_aspect::execution_timestamps);
    } else if(asp == aspect::usm_device_allocations) {
      return get_rt_device()->has(
          rt::device_support_aspect::usm_device_allocations);
    } else if(asp == aspect::usm_host_allocations) {
      return get_rt_device()->has(
          rt::device_support_aspect::usm_host_allocations);
    } else if(asp == aspect::usm_atomic_host_allocations) {
      return get_rt_device()->has(
          rt::device_support_aspect::usm_atomic_host_allocations);
    } else if(asp == aspect::usm_shared_allocations) {
      return get_rt_device()->has(
          rt::device_support_aspect::usm_shared_allocations);
    } else if(asp == aspect::usm_system_allocations) {
      return get_rt_device()->has(
          rt::device_support_aspect::usm_system_allocations);
    }

    return false;
  }

  bool hipSYCL_has_compiled_kernels() const {
#if defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
    if (is_cpu())
      return true;
#endif
    
#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
    if(_device_id.get_backend() == rt::backend_id::cuda)
      return true;
#endif
    
#if defined(__HIPSYCL_ENABLE_HIP_TARGET__)
    if(_device_id.get_backend() == rt::backend_id::hip)
      return true;
#endif

#if defined(__HIPSYCL_ENABLE_SPIRV_TARGET__)
    if(_device_id.get_backend() == rt::backend_id::level_zero)
      return true;
#endif
    
    return false;
  }

  // Implemented in platform.hpp
  platform get_platform() const;

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const;

  bool has_extension(const string_class &extension) const
  {
    return false;
  }



  // Available only when prop == info::partition_property::partition_equally
  template <info::partition_property prop,
            std::enable_if_t<prop == info::partition_property::partition_equally>*
              = nullptr>
  std::vector<device> create_sub_devices(size_t nbSubDev) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop,
            std::enable_if_t<prop == info::partition_property::partition_by_counts>*
              = nullptr>
  std::vector<device> create_sub_devices(const std::vector<size_t> &counts) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }

  // Available only when prop == info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop,
            std::enable_if_t<prop == info::partition_property::partition_by_affinity_domain>*
              = nullptr>
  std::vector<device> create_sub_devices(info::partition_affinity_domain
                                          affinityDomain) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }

  static std::vector<device>
  get_devices(info::device_type deviceType = info::device_type::all) {

    std::vector<device> result;

    rt::application::backends().for_each_backend(
        [&](rt::backend *b) {
          rt::backend_descriptor bd = b->get_backend_descriptor();
          std::size_t num_devices =
              b->get_hardware_manager()->get_num_devices();

          for (std::size_t dev = 0; dev < num_devices; ++dev) {
            rt::device_id d_id{bd, static_cast<int>(dev)};

            device d;
            d._device_id = d_id;

            if (deviceType == info::device_type::all ||
                (deviceType == info::device_type::accelerator &&
                 d.is_accelerator()) ||
                (deviceType == info::device_type::cpu && d.is_cpu()) ||
                (deviceType == info::device_type::host && d.is_cpu()) ||
                (deviceType == info::device_type::gpu && d.is_gpu())) {

              result.push_back(d);
            }
          }
        });

    return result;
  }

  static int get_num_devices() {
    return get_devices(info::device_type::all).size();
  }

  friend bool operator ==(const device& lhs, const device& rhs)
  { return lhs._device_id == rhs._device_id; }

  friend bool operator!=(const device& lhs, const device &rhs)
  { return !(lhs == rhs); }
  
  backend get_backend() const noexcept {
    return _device_id.get_backend();
  }
private:
  rt::device_id _device_id;

  rt::hardware_context *get_rt_device() const {
    auto ptr = rt::application::get_backend(_device_id.get_backend())
                   .get_hardware_manager()
                   ->get_device(_device_id.get_id());
    if (!ptr) {
      throw runtime_error{"Could not access device"};
    }
    return ptr;
  }
};

HIPSYCL_SPECIALIZE_GET_INFO(device, device_type) {
  if (this->is_cpu())
    return info::device_type::cpu;
  else if (this->is_gpu())
    return info::device_type::gpu;
  else
    return info::device_type::custom;
}

/// \todo Return different id for amd and nvidia
HIPSYCL_SPECIALIZE_GET_INFO(device, vendor_id)
{ 
  return get_rt_device()->get_property(
      rt::device_uint_property::vendor_id); 
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_compute_units)
{
  return get_rt_device()->get_property(
      rt::device_uint_property::max_compute_units);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_item_dimensions)
{ return 3; }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_item_sizes)
{
  std::size_t size0 = static_cast<std::size_t>(get_rt_device()->get_property(
      rt::device_uint_property::max_global_size0));
  std::size_t size1 = static_cast<std::size_t>(get_rt_device()->get_property(
      rt::device_uint_property::max_global_size1));
  std::size_t size2 = static_cast<std::size_t>(get_rt_device()->get_property(
      rt::device_uint_property::max_global_size2));
  return id<3>{size0, size1, size2};
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_group_size)
{
  return static_cast<size_t>(
      get_rt_device()->get_property(rt::device_uint_property::max_group_size));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_num_sub_groups)
{
  return static_cast<unsigned int>(
      get_rt_device()->get_property(rt::device_uint_property::max_num_sub_groups));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, sub_group_independent_forward_progress)
{
  return get_rt_device()->has(
      rt::device_support_aspect::sub_group_independent_forward_progress);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, sub_group_sizes)
{
  return get_rt_device()->get_property(
      rt::device_uint_list_property::sub_group_sizes);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_char) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_char));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_double){
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_double));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_float) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_float));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_half) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_half));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_int) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_int));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_long) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_long));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_short) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::preferred_vector_width_short));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_char) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_char));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_double) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_double));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_float) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_float));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_half) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_half));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_int) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_int));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_long) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_long));
}
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_short) {
  return static_cast<int>(get_rt_device()->get_property(
      rt::device_uint_property::native_vector_width_short));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_clock_frequency)
{
  return static_cast<detail::u_int>(
      get_rt_device()->get_property(rt::device_uint_property::max_clock_speed));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, address_bits) {
  return get_rt_device()->get_property(rt::device_uint_property::address_bits);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_mem_alloc_size)
{
  return static_cast<detail::u_long>(
      get_rt_device()->get_property(rt::device_uint_property::max_malloc_size));
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image_support) {
  return get_rt_device()->has(rt::device_support_aspect::images);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_read_image_args) {
  return get_rt_device()->get_property(
      rt::device_uint_property::max_read_image_args);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_write_image_args) {
  return get_rt_device()->get_property(
      rt::device_uint_property::max_write_image_args);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image2d_max_width) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image2d_max_width);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image2d_max_height) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image2d_max_height);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_width) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image3d_max_width);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_height) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image3d_max_height);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_depth) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image3d_max_depth);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image_max_buffer_size) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image_max_buffer_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image_max_array_size) {
  return get_rt_device()->get_property(
      rt::device_uint_property::image_max_array_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_samplers) {
  return get_rt_device()->get_property(rt::device_uint_property::max_samplers);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_parameter_size) {
  return get_rt_device()->get_property(
      rt::device_uint_property::max_parameter_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, mem_base_addr_align) {
  return get_rt_device()->get_property(
      rt::device_uint_property::mem_base_addr_align);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, half_fp_config)
{
  return std::vector<info::fp_config>{
    info::fp_config::denorm,
    info::fp_config::inf_nan,
    info::fp_config::round_to_nearest,
    info::fp_config::round_to_zero,
    info::fp_config::round_to_inf,
    info::fp_config::fma,
    info::fp_config::correctly_rounded_divide_sqrt
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, single_fp_config)
{
  return std::vector<info::fp_config>{
    info::fp_config::denorm,
    info::fp_config::inf_nan,
    info::fp_config::round_to_nearest,
    info::fp_config::round_to_zero,
    info::fp_config::round_to_inf,
    info::fp_config::fma,
    info::fp_config::correctly_rounded_divide_sqrt
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, double_fp_config)
{
  return std::vector<info::fp_config>{
    info::fp_config::denorm,
    info::fp_config::inf_nan,
    info::fp_config::round_to_nearest,
    info::fp_config::round_to_zero,
    info::fp_config::round_to_inf,
    info::fp_config::fma,
    info::fp_config::correctly_rounded_divide_sqrt
  };
}


HIPSYCL_SPECIALIZE_GET_INFO(device, name)
{
  return get_rt_device()->get_device_name();
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_type)
{
  if (!get_rt_device()->has(rt::device_support_aspect::global_mem_cache))
    return info::global_mem_cache_type::none;
  if (get_rt_device()->has(
          rt::device_support_aspect::global_mem_cache_read_only))
    return info::global_mem_cache_type::read_only;
  else if (get_rt_device()->has(
          rt::device_support_aspect::global_mem_cache_write_only))
    return info::global_mem_cache_type::write_only;

  return info::global_mem_cache_type::none;
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_line_size) {
  return get_rt_device()->get_property(
      rt::device_uint_property::global_mem_cache_line_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_size)
{
  return get_rt_device()->get_property(
      rt::device_uint_property::global_mem_cache_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_size)
{
  return get_rt_device()->get_property(
      rt::device_uint_property::global_mem_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_constant_buffer_size)
{
  return get_rt_device()->get_property(
      rt::device_uint_property::max_constant_buffer_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_constant_args) {
  return get_rt_device()->get_property(
      rt::device_uint_property::max_constant_args);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, local_mem_type) {
  if (get_rt_device()->has(rt::device_support_aspect::emulated_local_memory)) {
    return info::local_mem_type::global;
  } else {
    return info::local_mem_type::local;
  }
}

HIPSYCL_SPECIALIZE_GET_INFO(device, local_mem_size)
{
  return get_rt_device()->get_property(
      rt::device_uint_property::local_mem_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, error_correction_support) {
  return get_rt_device()->has(rt::device_support_aspect::error_correction);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, host_unified_memory) {
  return get_rt_device()->has(rt::device_support_aspect::host_unified_memory);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, profiling_timer_resolution)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_endian_little)
{ return get_rt_device()->has(rt::device_support_aspect::little_endian); }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_compiler_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_linker_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, execution_capabilities)
{
  return std::vector<info::execution_capability>{
    info::execution_capability::exec_kernel
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, queue_profiling) {
  return get_rt_device()->has(rt::device_support_aspect::execution_timestamps);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, built_in_kernels)
{ return std::vector<string_class>{}; }


HIPSYCL_SPECIALIZE_GET_INFO(device, vendor) {
  return get_rt_device()->get_vendor_name();
}

HIPSYCL_SPECIALIZE_GET_INFO(device, driver_version)
{
  return get_rt_device()->get_driver_version();
}

HIPSYCL_SPECIALIZE_GET_INFO(device, profile)
{ return get_rt_device()->get_profile(); }

HIPSYCL_SPECIALIZE_GET_INFO(device, version) {
  return "1.2 "+detail::version_string();
}

HIPSYCL_SPECIALIZE_GET_INFO(device, opencl_c_version)
{ return "1.2 HIPSYCL"; }

HIPSYCL_SPECIALIZE_GET_INFO(device, aspects)
{
  std::array aspects = {aspect::cpu,
                        aspect::gpu,
                        aspect::accelerator,
                        aspect::custom,
                        aspect::emulated,
                        aspect::host_debuggable,
                        aspect::fp16,
                        aspect::fp64,
                        aspect::atomic64,
                        aspect::image,
                        aspect::online_compiler,
                        aspect::online_linker,
                        aspect::queue_profiling,
                        aspect::usm_device_allocations,
                        aspect::usm_host_allocations,
                        aspect::usm_atomic_host_allocations,
                        aspect::usm_shared_allocations,
                        aspect::usm_atomic_shared_allocations,
                        aspect::usm_system_allocations};

  std::vector<aspect> result;
  
  for(auto asp : aspects) {
    if(this->has(asp)){
      result.push_back(asp);
    }
  }

  return result;
}

HIPSYCL_SPECIALIZE_GET_INFO(device, extensions)
{
  return std::vector<string_class>{};
}

HIPSYCL_SPECIALIZE_GET_INFO(device, printf_buffer_size) {
  return get_rt_device()->get_property(
      rt::device_uint_property::printf_buffer_size);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_interop_user_sync)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, parent_device)
{ throw invalid_object_error{"Device is not a subdevice"}; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_max_sub_devices) {
  return get_rt_device()->get_property(
      rt::device_uint_property::partition_max_sub_devices);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_properties)
{ return std::vector<info::partition_property>{}; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_affinity_domains)
{
  return std::vector<info::partition_affinity_domain>{
    info::partition_affinity_domain::not_applicable
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_type_property)
{ return info::partition_property::no_partition; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_type_affinity_domain)
{ return info::partition_affinity_domain::not_applicable; }


HIPSYCL_SPECIALIZE_GET_INFO(device, reference_count)
{
  // hipSYCL device classes do not need any resources, and hence
  // no reference counting is required.
  return 1;
}

namespace detail {

inline rt::device_id extract_rt_device(const device &d) {
  return d._device_id;
}

}

} // namespace sycl
} // namespace hipsycl



#endif
