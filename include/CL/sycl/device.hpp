/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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


#ifndef SYCU_DEVICE_HPP
#define SYCU_DEVICE_HPP

#include <limits>
#include <type_traits>

#include "types.hpp"
#include "info/info.hpp"
#include "backend/backend.hpp"
#include "exception.hpp"
#include "id.hpp"

namespace cl {
namespace sycl {

class device_selector;
class platform;

class device
{
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

  // OpenCL interop is not supported
  // explicit device(cl_device_id deviceId);

  explicit device(const device_selector &deviceSelector);

  // OpenCL interop is not supported
  // cl_device_id get() const;

  bool is_host() const {return false;}

  bool is_cpu() const {return false; }

  bool is_gpu() const {return true; }

  bool is_accelerator() const {return true; }

  platform get_platform() const;

  template <info::device param>
  typename info::param_traits<info::device, param>::return_type
  get_info() const
  {
    throw unimplemented{"device::get_info() is unimplemented"};
  }

  bool has_extension(const string_class &extension) const
  {
    throw unimplemented{"device::has_extension is unimplemented"};
  }

  /* create_sub_devices is not yet supported

  // Available only when prop == info::partition_property::partition_equally
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(size_t nbSubDev) const;

  // Available only when prop == info::partition_property::partition_by_counts
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(const vector_class<size_t> &counts) const
  {
    throw unimplemented{"device::create_sub_devices is unimplemented"};
  }

  // Available only when prop == info::partition_property::partition_by_affinity_domain
  template <info::partition_property prop>
  vector_class<device> create_sub_devices(info::affinity_domain affinityDomain) const
  {
    throw unimplemented{"device::create_sub_devices is unimplemented"};
  }
  */

  static vector_class<device> get_devices(
      info::device_type deviceType = info::device_type::all);

  static int get_num_devices();

  int get_device_id() const;

  bool operator ==(const device& rhs) const
  { return rhs._device_id == _device_id; }

  bool operator !=(const device& rhs) const
  { return !(*this == rhs); }
private:
  int _device_id;
};

SYCU_SPECIALIZE_GET_INFO(device, device_type)
{ return info::device_type::gpu; }

/// \todo Return different id for amd and nvidia
SYCU_SPECIALIZE_GET_INFO(device, vendor_id)
{ return 1; }

SYCU_SPECIALIZE_GET_INFO(device, max_compute_units)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_uint>(props.multiProcessorCount);
}

SYCU_SPECIALIZE_GET_INFO(device, max_work_item_dimensions)
{ return 3; }

SYCU_SPECIALIZE_GET_INFO(device, max_work_item_sizes)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return id<3>{
    static_cast<size_t>(props.maxThreadsDim[0]),
    static_cast<size_t>(props.maxThreadsDim[1]),
    static_cast<size_t>(props.maxThreadsDim[2])
  };
}

SYCU_SPECIALIZE_GET_INFO(device, max_work_group_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<size_t>(props.maxThreadsPerBlock);
}

SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_char)
{ return 4; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_double)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_float)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_half)
{ return 0; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_int)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_long)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, preferred_vector_width_short)
{ return 2; }


SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_char)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_double)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_float)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_half)
{ return 0; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_int)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_long)
{ return 1; }
SYCU_SPECIALIZE_GET_INFO(device, native_vector_width_short)
{ return 1; }

SYCU_SPECIALIZE_GET_INFO(device, max_clock_frequency)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_uint>(props.clockRate / 1000);
}

SYCU_SPECIALIZE_GET_INFO(device, address_bits)
{ return 64; }

SYCU_SPECIALIZE_GET_INFO(device, max_mem_alloc_size)
{
  // return global memory size for now
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_ulong>(props.totalGlobalMem);
}

SYCU_SPECIALIZE_GET_INFO(device, image_support)
{ return true; }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, max_read_image_args)
{ return 128; }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, max_write_image_args)
{ return 128; }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image2d_max_width)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image2d_max_height)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image3d_max_width)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image3d_max_height)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image3d_max_depth)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image_max_buffer_size)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, image_max_array_size)
{ return std::numeric_limits<size_t>::max(); }

/// \todo Find out actual value
SYCU_SPECIALIZE_GET_INFO(device, max_samplers)
{ return std::numeric_limits<cl_uint>::max(); }

SYCU_SPECIALIZE_GET_INFO(device, max_parameter_size)
{ return std::numeric_limits<size_t>::max(); }

SYCU_SPECIALIZE_GET_INFO(device, mem_base_addr_align)
{ return 8; }

SYCU_SPECIALIZE_GET_INFO(device, half_fp_config)
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

SYCU_SPECIALIZE_GET_INFO(device, single_fp_config)
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

SYCU_SPECIALIZE_GET_INFO(device, double_fp_config)
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


SYCU_SPECIALIZE_GET_INFO(device, name)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return string_class{props.name};
}

SYCU_SPECIALIZE_GET_INFO(device, global_mem_cache_type)
{
  return info::global_mem_cache_type::read_only;
}

/// \todo what is the cache line size on AMD devices?
SYCU_SPECIALIZE_GET_INFO(device, global_mem_cache_line_size)
{ return 128; }

SYCU_SPECIALIZE_GET_INFO(device, global_mem_cache_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_ulong>(props.l2CacheSize);
}

SYCU_SPECIALIZE_GET_INFO(device, global_mem_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_ulong>(props.totalGlobalMem);
}

SYCU_SPECIALIZE_GET_INFO(device, max_constant_buffer_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_ulong>(props.totalConstMem);
}

SYCU_SPECIALIZE_GET_INFO(device, max_constant_args)
{ return std::numeric_limits<cl_uint>::max(); }

SYCU_SPECIALIZE_GET_INFO(device, local_mem_type)
{ return info::local_mem_type::local; }

SYCU_SPECIALIZE_GET_INFO(device, local_mem_size)
{
  hipDeviceProp_t props;
  detail::check_error(hipGetDeviceProperties(&props, _device_id));
  return static_cast<cl_ulong>(props.sharedMemPerBlock);
}

/// \todo actually check support
SYCU_SPECIALIZE_GET_INFO(device, error_correction_support)
{ return false; }

SYCU_SPECIALIZE_GET_INFO(device, host_unified_memory)
{ return false; }

SYCU_SPECIALIZE_GET_INFO(device, profiling_timer_resolution)
{ return 1; }

SYCU_SPECIALIZE_GET_INFO(device, is_endian_little)
{ return true; }

SYCU_SPECIALIZE_GET_INFO(device, is_available)
{ return true; }

SYCU_SPECIALIZE_GET_INFO(device, is_compiler_available)
{ return true; }

SYCU_SPECIALIZE_GET_INFO(device, is_linker_available)
{ return true; }

SYCU_SPECIALIZE_GET_INFO(device, execution_capabilities)
{
  return vector_class<info::execution_capability>{
    info::execution_capability::exec_kernel
  };
}

SYCU_SPECIALIZE_GET_INFO(device, queue_profiling)
{ return false; }

SYCU_SPECIALIZE_GET_INFO(device, built_in_kernels)
{ return vector_class<string_class>{}; }


SYCU_SPECIALIZE_GET_INFO(device, vendor)
{
#ifdef SYCU_PLATFORM_CUDA
  return string_class{"NVIDIA"};
#else
  return string_class{"AMD"};
#endif
}

SYCU_SPECIALIZE_GET_INFO(device, driver_version)
{
  int version;
  detail::check_error(hipRuntimeGetVersion(&version));
  return std::to_string(version)+".0";
}

SYCU_SPECIALIZE_GET_INFO(device, profile)
{ return "SYCU/CUDA/HIP"; }

SYCU_SPECIALIZE_GET_INFO(device, version)
{ return "1.2"; }

SYCU_SPECIALIZE_GET_INFO(device, opencl_c_version)
{ return "1.2"; }

SYCU_SPECIALIZE_GET_INFO(device, extensions)
{
  return vector_class<string_class>{};
}

SYCU_SPECIALIZE_GET_INFO(device, printf_buffer_size)
{ return std::numeric_limits<size_t>::max(); }

SYCU_SPECIALIZE_GET_INFO(device, preferred_interop_user_sync)
{ return true; }

SYCU_SPECIALIZE_GET_INFO(device, parent_device)
{ throw invalid_object_error{"Device is not a subdevice"}; }

SYCU_SPECIALIZE_GET_INFO(device, partition_max_sub_devices)
{ return 0; }

SYCU_SPECIALIZE_GET_INFO(device, partition_properties)
{ return vector_class<info::partition_property>{}; }

SYCU_SPECIALIZE_GET_INFO(device, partition_affinity_domains)
{
  return vector_class<info::partition_affinity_domain>{
    info::partition_affinity_domain::not_applicable
  };
}

SYCU_SPECIALIZE_GET_INFO(device, partition_type_property)
{ return info::partition_property::no_partition; }

SYCU_SPECIALIZE_GET_INFO(device, partition_type_affinity_domain)
{ return info::partition_affinity_domain::not_applicable; }


SYCU_SPECIALIZE_GET_INFO(device, reference_count)
{
  // SYCU device classes do not need any resources, and hence
  // no reference counting is required.
  return 1;
}

namespace detail {

void set_device(const device& d);

}


} // namespace sycl
} // namespace cl



#endif
