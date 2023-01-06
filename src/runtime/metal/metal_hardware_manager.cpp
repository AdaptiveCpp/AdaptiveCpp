/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#include <cassert>
#include <cstdint>
#include <string>
#include <limits>

// TODO: Make a special file to declare these implementations.
// You must define these *before* including Metal.hpp.
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>
#include <sys/sysctl.h>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/metal/metal_hardware_manager.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

inline io_registry_entry_t get_gpu_entry();
inline int64_t get_gpu_core_count(io_registry_entry_t gpu_entry);
inline int64_t get_gpu_max_clock_speed(io_registry_entry_t gpu_entry);
inline int64_t get_gpu_slc_size(io_registry_entry_t gpu_entry);
inline int64_t get_max_allocated_size(MTL::Device* device);

metal_hardware_context::metal_hardware_context(MTL::Device* device)
  : _device{device}
{
  // Retain to prevent the caller from making this a zombie object.
  device->retain();
  
  if (device->supportsFamily(MTL::GPUFamily(MTL::GPUFamilyApple8 + 1))) {
    // Detect future GPU families and handle them specially.
    int64_t highest_family = MTL::GPUFamilyApple8 + 1;
    while (true) {
      if (device->supportsFamily(MTL::GPUFamily(highest_family + 1))) {
        continue;
      } else {
        this->_gpu_family = MTL::GPUFamily(highest_family);
      }
    }
  } else if (device->supportsFamily(MTL::GPUFamilyApple8)) {
    this->_gpu_family = MTL::GPUFamilyApple8;
  } else if (device->supportsFamily(MTL::GPUFamilyApple7)) {
    this->_gpu_family = MTL::GPUFamilyApple7;
  } else {
    // TODO: Is there a way to fail at runtime without printing a nonexistent
    // error code? If so, clean up the error message here and in other functions
    // for getting the GPU entry.
    print_error(__hipsycl_here(),
                error_info{"metal_hardware_context: Device does not support GPUFamilyApple7",
      error_code{"metal", 0}});
  }
  
  // Fetch expensive properties beforehand, instead of lazily caching them.
  io_registry_entry_t gpu_entry = get_gpu_entry();
  this->_core_count = get_gpu_core_count(gpu_entry);
  this->_max_clock_speed = get_gpu_max_clock_speed(gpu_entry);
  this->_slc_size = get_gpu_slc_size(gpu_entry);
  IOObjectRelease(gpu_entry);
  
  this->_max_allocated_size = get_max_allocated_size(_device);
}

bool metal_hardware_context::is_cpu() const {
  return false;
}

bool metal_hardware_context::is_gpu() const {
  return true;
}

// We still don't know whether 4x concurrency is specific to the A15, A-series
// variants in general, or the Apple 8 generation.
// Source: https://github.com/philipturner/metal-benchmarks
std::size_t metal_hardware_context::get_max_kernel_concurrency() const {
  if (_gpu_family == MTL::GPUFamilyApple7) {
    return _core_count * 3;
  } else if (_gpu_family == MTL::GPUFamilyApple8) {
    return _core_count * 4;
  } else {
    // Return a default for unrecognized models.
    return _core_count * 4;
  }
}

// Equals kernel concurrency because we implement blits through custom compute
// kernels.
std::size_t metal_hardware_context::get_max_memcpy_concurrency() const {
  return get_max_kernel_concurrency();
}

std::string metal_hardware_context::get_device_name() const {
  return std::string(_device->name()->cString(NS::UTF8StringEncoding));
}

std::string metal_hardware_context::get_vendor_name() const {
  return "Apple";
}

// We return the Metal GPU family instead of the actual architecture name (e.g.
// g13p, g13d, g14p) because it's more useful. Also, it's hard to perfectly
// infer the number after "g" on future hardware. It may not completely align
// with the number on "Apple Mx".
std::string metal_hardware_context::get_device_arch() const {
  if (_gpu_family == MTL::GPUFamilyApple7) {
    return "apple7";
  } else if (_gpu_family == MTL::GPUFamilyApple8) {
    return "apple8";
  } else {
    int64_t number = 8 + (_gpu_family - MTL::GPUFamilyApple8);
    return std::string("apple") + std::to_string(number);
  }
}

bool metal_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return false;
    break;
  case device_support_aspect::host_unified_memory:
    return true;
    break;
  case device_support_aspect::error_correction:
    return false;
    break;
  case device_support_aspect::global_mem_cache:
    return true;
    break;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
    break;
  case device_support_aspect::global_mem_cache_write_only:
    return false;
    break;
  case device_support_aspect::images:
    return false;
    break;
  case device_support_aspect::little_endian:
    return true;
    break;
  case device_support_aspect::sub_group_independent_forward_progress:
    return true;
    break;
  case device_support_aspect::usm_device_allocations:
    return true;
    break;
  case device_support_aspect::usm_host_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_host_allocations:
    return false;
    break;
  case device_support_aspect::usm_shared_allocations:
    return true;
    break;
  case device_support_aspect::usm_atomic_shared_allocations:
    return false;
    break;
  case device_support_aspect::usm_system_allocations:
    return false;
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
  std::terminate();
}

std::size_t metal_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return _core_count;
    break;
  case device_uint_property::max_global_size0:
    return int64_t(__UINT32_MAX__) + 1;
    break;
  case device_uint_property::max_global_size1:
    return int64_t(__UINT32_MAX__) + 1;
    break;
  case device_uint_property::max_global_size2:
    return int64_t(__UINT32_MAX__) + 1;
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
    return 1024 / 32;
    break;
  case device_uint_property::preferred_vector_width_char:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_double:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_float:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_half:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_int:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_long:
    return 1;
    break;
  case device_uint_property::preferred_vector_width_short:
    return 1;
    break;
  case device_uint_property::native_vector_width_char:
    return 1;
    break;
  case device_uint_property::native_vector_width_double:
    return 1;
    break;
  case device_uint_property::native_vector_width_float:
    return 1;
    break;
  case device_uint_property::native_vector_width_half:
    return 1;
    break;
  case device_uint_property::native_vector_width_int:
    return 1;
    break;
  case device_uint_property::native_vector_width_long:
    return 1;
    break;
  case device_uint_property::native_vector_width_short:
    return 1;
    break;
  case device_uint_property::max_clock_speed:
    return _max_clock_speed;
    break;
  case device_uint_property::max_malloc_size:
    return _device->maxBufferLength();
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
    // This is the M1 CPU's cache line size, a common size on GPUs, and equal
    // to SIMD size (32) times word size (4). However, 64 B was another magic
    // number when creating the custom blit command encoder. Perhaps that's
    // because it equals the RAM bus width on M1 Max.
    return 128;
    break;
  case device_uint_property::global_mem_cache_size:
    return _slc_size;
    break;
  case device_uint_property::global_mem_size:
    return _max_allocated_size;
    break;
  case device_uint_property::max_constant_buffer_size:
    return 4096;
    break;
  case device_uint_property::max_constant_args:
    // Metal supports at most 31 buffers bound into the argument table (the
    // only place for constant arguments). If you exceed that amount, we place
    // the rest somewhere else (not actually constant). Constant is just a
    // suggestion anyway; it only restricts data copied by value through
    // MTLDevice.setBytes(_:offset:index).
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::local_mem_size:
    return 32 * 1024;
    break;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
    break;
  case device_uint_property::partition_max_sub_devices:
    return 0;
    break;
  case device_uint_property::vendor_id:
    // The device ID returned by OpenCL on macOS. This is the same across both
    // the M1 and M1 Max, so likely universal.
    return 0x1027f00;
    break;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::vector<std::size_t>
metal_hardware_context::get_property(device_uint_list_property prop) const {
  switch(prop) {
  case device_uint_list_property::sub_group_sizes:
    std::vector<std::size_t> result(1);
    result[0] = 32;
    return result;
    break;
  }

  assert(false && "Invalid device property");
  std::terminate();
}

// Return the Metal Shading Language version, which coincides with the marketed
// Metal version.
std::string metal_hardware_context::get_driver_version() const {
  // TODO: Allow for multiple supported Metal versions, when the OS after macOS
  // Ventura comes out.
  return "3.0";
}

std::string metal_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

metal_hardware_context::~metal_hardware_context() {
  // Return the device's reference count to its original value.
  _device->release();
}

metal_hardware_manager::metal_hardware_manager() {
  // There should be only one device.
  NS::Array* all_devices = MTL::CopyAllDevices();
  NS::UInteger num_devices = all_devices->count();
  
  for (NS::UInteger i = 0; i < num_devices; ++i) {
    MTL::Device* mtl_device = (MTL::Device*)all_devices->object(i);
    metal_hardware_context hardware_context = metal_hardware_context{mtl_device};
    this->_devices.push_back(hardware_context);
  }
}

std::size_t metal_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *metal_hardware_manager::get_device(std::size_t index) {
  assert(index < _devices.size());
  return &(_devices[index]);
}

device_id metal_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{backend_descriptor{
    hardware_platform::metal, api_platform::metal},
    static_cast<int>(index)
  };
}

}
}

// TODO: Find a better way to visually separate helper functions from public
// API functions. We don't typically break and re-declare a namespace block.

namespace hipsycl {
namespace rt {

// The caller must release the entry.
inline io_registry_entry_t get_gpu_entry() {
  // Class hierarchy: IOGPU -> AGXAccelerator -> AGXFamilyAccelerator
  // We could go with IOGPU, but we want to restrict this to Apple silicon.
  CFMutableDictionaryRef match_dictionary = IOServiceMatching("AGXAccelerator");
  if (!match_dictionary) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_entry: Could not find AGXAccelerator service",
      error_code{"metal", 0}});
  }
  
  // Get the GPU's entry object.
  io_iterator_t entry_iterator;
  kern_return_t error = IOServiceGetMatchingServices(
    kIOMainPortDefault, match_dictionary, &entry_iterator);
  if (error != kIOReturnSuccess) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_entry: No objects match AGXAccelerator service",
      error_code{"metal", 0}});
  }
  io_registry_entry_t gpu_entry = IOIteratorNext(entry_iterator);
  if (IOIteratorNext(entry_iterator)) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_entry: Found multiple GPUs",
      error_code{"metal", 0}});
  }
  
  // Release acquired objects.
  IOObjectRelease(entry_iterator);
  return gpu_entry;
}

// Number of GPU cores.
inline int64_t get_gpu_core_count(io_registry_entry_t gpu_entry) {
#if TARGET_OS_IPHONE
  // TODO: Determine the core count on iOS through something like DeviceKit.
#else
  // Get the number of cores.
  CFNumberRef gpu_core_count = (CFNumberRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
  if (!gpu_core_count) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_core_count: Could not find 'gpu-core-count' property",
      error_code{"metal", 0}});
  }
  CFNumberType type = CFNumberGetType(gpu_core_count);
  if (type != kCFNumberSInt64Type) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_core_count: 'gpu-core-count' not type sInt64",
      error_code{"metal", 0}});
  }
  int64_t value;
  bool retrieved_value = CFNumberGetValue(gpu_core_count, type, &value);
  if (!retrieved_value) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_core_count: Could not fetch 'gpu-core-count' value",
      error_code{"metal", 0}});
  }
  
  // Release acquired objects.
  CFRelease(gpu_core_count);
  return value;
#endif
}

// Clock speed in MHz.
inline int64_t get_gpu_max_clock_speed(io_registry_entry_t gpu_entry) {
  CFStringRef model = (CFStringRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("model"), kCFAllocatorDefault, 0);
  if (!model) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_max_clock_speed: Could not find 'model' property",
      error_code{"metal", 0}});
  }
  
  // Newest data on each model's clock speed are located at:
  // https://github.com/philipturner/metal-benchmarks
  if (CFStringHasPrefix(model, CFSTR("Apple M1"))) {
    if (CFStringHasSuffix(model, CFSTR("M1"))) {
      return 1278;
    } else if (CFStringHasSuffix(model, CFSTR("Pro"))) {
      return 1296;
    } else if (CFStringHasSuffix(model, CFSTR("Max"))) {
      return 1296;
    } else if (CFStringHasSuffix(model, CFSTR("Ultra"))) {
      return 1296;
    } else {
      // Return a default for unrecognized models.
      return 1296;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple M2"))) {
    if (CFStringHasSuffix(model, CFSTR("M2"))) {
      return 1398;
    } else {
      // Return a default for unrecognized models.
      return 1398;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple M"))) {
    // Return a default for unrecognized models.
    return 1398;
  } else if (CFStringHasPrefix(model, CFSTR("Apple A"))) {
    if (CFStringHasSuffix(model, CFSTR("A14"))) {
      return 1278;
    } else if (CFStringHasSuffix(model, CFSTR("A15"))) {
      return 1336;
    } else if (CFStringHasSuffix(model, CFSTR("A16"))) {
      return 1336;
    } else {
      // Return a default for unrecognized models.
      return 1336;
    }
  } else {
    // Could not extract any information about the GPU.
    return 0;
  }
}

// Size of the largest data cache.
inline int64_t get_gpu_slc_size(io_registry_entry_t gpu_entry) {
  CFStringRef model = (CFStringRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("model"), kCFAllocatorDefault, 0);
  if (!model) {
    print_error(__hipsycl_here(),
                error_info{"get_gpu_max_clock_speed: Could not find 'model' property",
      error_code{"metal", 0}});
  }
  
  int64_t megabytes = 0;
  if (CFStringHasPrefix(model, CFSTR("Apple M1"))) {
    if (CFStringHasSuffix(model, CFSTR("M1"))) {
      megabytes = 8;
    } else if (CFStringHasSuffix(model, CFSTR("Pro"))) {
      megabytes = 24;
    } else if (CFStringHasSuffix(model, CFSTR("Max"))) {
      megabytes = 48;
    } else if (CFStringHasSuffix(model, CFSTR("Ultra"))) {
      megabytes = 96;
    } else {
      // Return a default for unrecognized models.
      megabytes = 96;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple M2")) &&
             CFStringHasSuffix(model, CFSTR("M2"))) {
    megabytes = 8;
  } else if (CFStringHasPrefix(model, CFSTR("Apple M"))) {
    // Return a default for unrecognized models.
    if (CFStringHasSuffix(model, CFSTR("Pro"))) {
      megabytes = 24;
    } else if (CFStringHasSuffix(model, CFSTR("Max"))) {
      megabytes = 48;
    } else if (CFStringHasSuffix(model, CFSTR("Ultra"))) {
      megabytes = 96;
    } else /*Likely base M-series model.*/ {
      megabytes = 8;
    }
  } else if (CFStringHasPrefix(model, CFSTR("Apple A"))) {
    if (CFStringHasSuffix(model, CFSTR("A14"))) {
      megabytes = 16;
    } else if (CFStringHasSuffix(model, CFSTR("A15"))) {
      megabytes = 32;
    } else if (CFStringHasSuffix(model, CFSTR("A16"))) {
      megabytes = 24;
    } else {
      // Return a default for unrecognized models.
      megabytes = 24;
    }
  } else {
    // Could not extract any information about the GPU.
    megabytes = 0;
  }
  return megabytes * 1024 * 1024;
}

// The maximum amount of VM memory you can materialize simultaneously.
inline int64_t get_max_allocated_size(MTL::Device* device) {
  int64_t system_memory = 0;
  size_t size = sizeof(system_memory);
  int error = sysctlbyname("hw.memsize", &system_memory, &size, NULL, 0);
  
  if (error) {
    print_error(__hipsycl_here(),
                error_info{"get_max_allocated_size: Could not find 'hw.memsize'",
      error_code{"metal", errno}});
  }
  
  // The tested limit is ~3725 / 5494 MB on iOS, 67% of physical RAM. It is
  // ~21700 / 32768 MB on macOS, 66% of physical RAM. We go slightly under this
  // limit (65%) for safety.
  // TODO: Test whether the limit ever breaks.
  int64_t megabytes = system_memory / 1024 / 1024;
  int64_t working_set_megabytes = 65 * megabytes / 100;
  int64_t working_set = working_set_megabytes * 1024 * 1024;
#if TARGET_OS_IPHONE
  // NOTE: On iOS, you must set increased-memory-limit and
  // extended-virtual-addressing in the app's entitlements file. Otherwise this
  // will crash at runtime.
  return working_set;
#else
  int64_t metal_reported_limit = device->recommendedMaxWorkingSetSize();
  return std::min(working_set, metal_reported_limit);
#endif
}

}
}
