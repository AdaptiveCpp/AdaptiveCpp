/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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


#include <omp.h>
#include <limits>

#include "hipSYCL/runtime/omp/omp_hardware_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"

namespace hipsycl {
namespace rt {


bool omp_hardware_context::is_cpu() const {
  return true;
}

bool omp_hardware_context::is_gpu() const {
  return false;
}

std::size_t omp_hardware_context::get_max_kernel_concurrency() const {
  return 1;
}
  
// TODO We could actually copy have more memcpy concurrency
std::size_t omp_hardware_context::get_max_memcpy_concurrency() const {
  return 1;
}

std::string omp_hardware_context::get_device_name() const {
  return "hipSYCL OpenMP host device";
}

std::string omp_hardware_context::get_vendor_name() const {
  return "the hipSYCL project";
}

std::string omp_hardware_context::get_device_arch() const {
  return "<native-cpu>";
}

bool omp_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return true;
    break;
  case device_support_aspect::host_unified_memory:
    return true;
    break;
  case device_support_aspect::error_correction:
    return false; // TODO: Actually query this
    break;
  case device_support_aspect::global_mem_cache:
    return true;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
    break;
  case device_support_aspect::global_mem_cache_read_write:
    return true;
    break;
  case device_support_aspect::images:
    return false;
    break;
  case device_support_aspect::little_endian:
#if defined(LITTLE_ENDIAN) || defined(__LITTLE_ENDIAN__) ||                    \
    defined(__ORDER_LITTLE_ENDIAN__)
    return true;
#else
    return false;
#endif
    break;
  case device_support_aspect::sub_group_independent_forward_progress:
    return false;
    break;
  case device_support_aspect::usm_device_allocations:
    return true;
    break;
  case device_support_aspect::usm_host_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_host_allocations:
    return true;
    break;
  case device_support_aspect::usm_shared_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_shared_allocations:
    return true;
    break;
  case device_support_aspect::usm_system_allocations:
    return true;
    break;
  case device_support_aspect::execution_timestamps:
    return true;
    break;
  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return true;
#else
    return false;
#endif
    break;
  }
  assert(false && "Unknown device aspect");
  return false;
}

std::size_t
omp_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return omp_get_num_procs();
    break;
  case device_uint_property::max_global_size0:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::max_global_size1:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::max_global_size2:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::max_group_size0:
    return 1024;
    break;
  case device_uint_property::max_group_size1:
    return 1024;
    break;
  case device_uint_property::max_group_size2:
    return 1024;
    break;
  case device_uint_property::max_group_size:
    return 1024;
    break;
  case device_uint_property::max_num_sub_groups:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::needs_dimension_flip:
    return true;
    break;
  case device_uint_property::preferred_vector_width_char:
    return 4;
    break;
  case device_uint_property::preferred_vector_width_double:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_float:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_half:
    return 2;
    break;
  case device_uint_property::preferred_vector_width_int:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_long:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_short:
    return 2;
    break;
  case device_uint_property::native_vector_width_char:
    return 4;
    break;
  case device_uint_property::native_vector_width_double:
    return 1;
    break;
  case device_uint_property::native_vector_width_float:
    return 1;
    break;
  case device_uint_property::native_vector_width_half:
    return 2;
    break;
  case device_uint_property::native_vector_width_int:
    return 1;
    break;
  case device_uint_property::native_vector_width_long:
    return 1;
    break;
  case device_uint_property::native_vector_width_short:
    return 2;
    break;
  case device_uint_property::max_clock_speed:
    return 0;
    break;
  case device_uint_property::max_malloc_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::address_bits:
    return 64;
    break;
  case device_uint_property::max_read_image_args:
    return 0;
    break;
  case device_uint_property::max_write_image_args:
    return 0;
    break;
  case device_uint_property::image2d_max_width:
    return 0;
    break;
  case device_uint_property::image2d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_width:
    return 0;
    break;
  case device_uint_property::image3d_max_height:
    return 0;
    break;
  case device_uint_property::image3d_max_depth:
    return 0;
    break;
  case device_uint_property::image_max_buffer_size:
    return 0;
    break;
  case device_uint_property::image_max_array_size:
    return 0;
    break;
  case device_uint_property::max_samplers:
    return 0;
    break;
  case device_uint_property::max_parameter_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::mem_base_addr_align:
    return 8; // TODO
    break;
  case device_uint_property::global_mem_cache_line_size:
    return 64; //TODO
    break;
  case device_uint_property::global_mem_cache_size:
    return 1; // TODO
    break;
  case device_uint_property::global_mem_size:
    return std::numeric_limits<std::size_t>::max(); // TODO
    break;
  case device_uint_property::max_constant_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::max_constant_args:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::local_mem_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  case device_uint_property::vendor_id:
    return std::numeric_limits<std::size_t>::max();
    break;
  }
  assert(false && "Invalid device property");
  return 0;
}

std::vector<std::size_t> omp_hardware_context::get_property(device_uint_list_property prop) const
{
  switch(prop) {
  case device_uint_list_property::sub_group_sizes:
    return std::vector<std::size_t>{1};
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::string omp_hardware_context::get_driver_version() const { return "1.2"; }

std::string omp_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

std::size_t omp_hardware_manager::get_num_devices() const { return 1; }


hardware_context* omp_hardware_manager::get_device(std::size_t index) {
  if(index != 0) {
    register_error(__acpp_here(),
                   error_info{"omp_hardware_manager: Requested device " +
                                  std::to_string(index) + " does not exist.",
                              error_type::invalid_parameter_error});
    return nullptr;
  }

  return &_device;
}

device_id omp_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{
      backend_descriptor{hardware_platform::cpu, api_platform::omp},
      static_cast<int>(index)};
}


}
}
