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

#include <limits>
#include <type_traits>

#include "types.hpp"
#include "info/info.hpp"
#include "backend/backend.hpp"
#include "exception.hpp"
#include "id.hpp"
#include "version.hpp"

namespace hipsycl {
namespace sycl {

class device_selector;
class platform;
class device;

namespace detail {
void set_device(const device& d);
}

class device
{
  friend void detail::set_device(const device&);
public:

  /// Since we do not support host execution, this will actually
  /// try to use the first GPU. Note: SYCL spec requires that
  /// this should actually create a device object for host execution.
  ///
  /// \todo Should this call throw an error instead of behaving differently
  /// than the spec requires?
  device()
    : _device_id{0}
  {}

#ifdef HIPSYCL_HIP_INTEROP
  device(int hipDeviceId)
    : _device_id{hipDeviceId}
  {}
#endif

  // OpenCL interop is not supported
  // explicit device(cl_device_id deviceId);

  explicit device(const device_selector &deviceSelector);

  // OpenCL interop is not supported
  // cl_device_id get() const;

  bool is_host() const 
  {
#ifdef HIPSYCL_PLATFORM_CPU
    return true;
#else
    return false;
#endif
  }

  bool is_cpu() const
  {
    return !is_gpu();
  }

  bool is_gpu() const
  {
#ifdef HIPSYCL_PLATFORM_CPU
    return false;
#else
    return true; 
#endif
  }

  bool is_accelerator() const 
  {
    return is_gpu(); 
  }

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
  vector_class<device> create_sub_devices(size_t nbSubDev) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop,
            std::enable_if_t<prop == info::partition_property::partition_by_counts>*
              = nullptr>
  vector_class<device> create_sub_devices(const vector_class<size_t> &counts) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }

  // Available only when prop == info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop,
            std::enable_if_t<prop == info::partition_property::partition_by_affinity_domain>*
              = nullptr>
  vector_class<device> create_sub_devices(info::partition_affinity_domain
                                          affinityDomain) const
  {
    throw feature_not_supported{"subdevices are unsupported."};
  }


  static vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all);

  static int get_num_devices();


#ifdef HIPSYCL_HIP_INTEROP
  int get_device_id() const;
#endif

  friend bool operator ==(const device& lhs, const device& rhs)
  { return rhs._device_id == lhs._device_id; }

  friend bool operator !=(const device& lhs, const device& rhs)
  { return !(lhs == rhs); }
private:
  int _device_id;
};

HIPSYCL_SPECIALIZE_GET_INFO(device, device_type)
{ return info::device_type::gpu; }

/// \todo Return different id for amd and nvidia
HIPSYCL_SPECIALIZE_GET_INFO(device, vendor_id)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_compute_units)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_int>(props.multiProcessorCount);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_item_dimensions)
{ return 3; }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_item_sizes)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return id<3>{
    static_cast<size_t>(props.maxThreadsDim[0]),
    static_cast<size_t>(props.maxThreadsDim[1]),
    static_cast<size_t>(props.maxThreadsDim[2])
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_work_group_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<size_t>(props.maxThreadsPerBlock);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_char)
{ return 4; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_double)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_float)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_half)
{ return 0; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_int)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_long)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_vector_width_short)
{ return 2; }


HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_char)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_double)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_float)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_half)
{ return 0; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_int)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_long)
{ return 1; }
HIPSYCL_SPECIALIZE_GET_INFO(device, native_vector_width_short)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_clock_frequency)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_int>(props.clockRate / 1000);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, address_bits)
{ return 64; }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_mem_alloc_size)
{
  // return global memory size for now
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_long>(props.totalGlobalMem);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, image_support)
{ return true; }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, max_read_image_args)
{ return 128; }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, max_write_image_args)
{ return 128; }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image2d_max_width)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image2d_max_height)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_width)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_height)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image3d_max_depth)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image_max_buffer_size)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, image_max_array_size)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
HIPSYCL_SPECIALIZE_GET_INFO(device, max_samplers)
{ return std::numeric_limits<detail::u_int>::max(); }

HIPSYCL_SPECIALIZE_GET_INFO(device, max_parameter_size)
{ return std::numeric_limits<size_t>::max(); }

HIPSYCL_SPECIALIZE_GET_INFO(device, mem_base_addr_align)
{ return 8; }

HIPSYCL_SPECIALIZE_GET_INFO(device, half_fp_config)
{
  return vector_class<info::fp_config>{
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
  return vector_class<info::fp_config>{
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
  return vector_class<info::fp_config>{
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
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return string_class{props.name};
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_type)
{
  return info::global_mem_cache_type::read_only;
}

/// \todo what is the cache line size on AMD devices?
HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_line_size)
{ return 128; }

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_cache_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_long>(props.l2CacheSize);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, global_mem_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_long>(props.totalGlobalMem);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_constant_buffer_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_long>(props.totalConstMem);
}

HIPSYCL_SPECIALIZE_GET_INFO(device, max_constant_args)
{ return std::numeric_limits<detail::u_int>::max(); }

HIPSYCL_SPECIALIZE_GET_INFO(device, local_mem_type)
{ return info::local_mem_type::local; }

HIPSYCL_SPECIALIZE_GET_INFO(device, local_mem_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<detail::u_long>(props.sharedMemPerBlock);
}

/// \todo actually check support
HIPSYCL_SPECIALIZE_GET_INFO(device, error_correction_support)
{ return false; }

HIPSYCL_SPECIALIZE_GET_INFO(device, host_unified_memory)
{
#ifdef HIPSYCL_PLATFORM_CPU
  return true;
#else
  return false; 
#endif
}

HIPSYCL_SPECIALIZE_GET_INFO(device, profiling_timer_resolution)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_endian_little)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_compiler_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, is_linker_available)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, execution_capabilities)
{
  return vector_class<info::execution_capability>{
    info::execution_capability::exec_kernel
  };
}

HIPSYCL_SPECIALIZE_GET_INFO(device, queue_profiling)
{ return false; }

HIPSYCL_SPECIALIZE_GET_INFO(device, built_in_kernels)
{ return vector_class<string_class>{}; }


HIPSYCL_SPECIALIZE_GET_INFO(device, vendor)
{
#ifdef HIPSYCL_PLATFORM_CUDA
  return string_class{"NVIDIA"};
#elif defined(HIPSYCL_PLATFORM_HCC)
  return string_class{"AMD"};
#else
  return string_class{"hipCPU"};
#endif
}

HIPSYCL_SPECIALIZE_GET_INFO(device, driver_version)
{
  return detail::version_string();
}

HIPSYCL_SPECIALIZE_GET_INFO(device, profile)
{ return "FULL_PROFILE"; }

HIPSYCL_SPECIALIZE_GET_INFO(device, version)
{
#ifdef HIPSYCL_PLATFORM_CUDA
  return "1.2 "+detail::version_string()+", running on NVIDIA CUDA";
#elif defined(HIPSYCL_PLATFORM_HCC)
  return "1.2 "+detail::version_string()+", running on AMD ROCm";
#else
  return "1.2 "+detail::version_string()+", running on hipCPU host device";
#endif
}

HIPSYCL_SPECIALIZE_GET_INFO(device, opencl_c_version)
{ return "1.2 HIPSYCL CUDA/HIP"; }

HIPSYCL_SPECIALIZE_GET_INFO(device, extensions)
{
  return vector_class<string_class>{};
}

HIPSYCL_SPECIALIZE_GET_INFO(device, printf_buffer_size)
{ return std::numeric_limits<size_t>::max(); }

HIPSYCL_SPECIALIZE_GET_INFO(device, preferred_interop_user_sync)
{ return true; }

HIPSYCL_SPECIALIZE_GET_INFO(device, parent_device)
{ throw invalid_object_error{"Device is not a subdevice"}; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_max_sub_devices)
{ return 0; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_properties)
{ return vector_class<info::partition_property>{}; }

HIPSYCL_SPECIALIZE_GET_INFO(device, partition_affinity_domains)
{
  return vector_class<info::partition_affinity_domain>{
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


} // namespace sycl
} // namespace hipsycl



#endif
