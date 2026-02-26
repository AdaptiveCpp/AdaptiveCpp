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
#include "hipSYCL/runtime/metal/metal_hardware_manager.hpp"

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION

#include <Metal/Metal.hpp>
#include <Foundation/Foundation.hpp>
#include <IOKit/IOKitLib.h>
#include <CoreFoundation/CoreFoundation.h>

#include <sys/sysctl.h>

namespace hipsycl {
namespace rt {

namespace {

// https://github.com/AdaptiveCpp/AdaptiveCpp/pull/901
// @philipturner provided these functions to query GPU properties

io_registry_entry_t get_gpu_entry() {
  // Class hierarchy: IOGPU -> AGXAccelerator -> AGXFamilyAccelerator
  // AGXAccelerator is only present on Apple Silicon; returns IO_OBJECT_NULL on
  // Intel Macs, virtual machines, and other non-Apple-Silicon environments.
  CFMutableDictionaryRef match_dictionary = IOServiceMatching("AGXAccelerator");
  if (!match_dictionary) {
    HIPSYCL_DEBUG_WARNING << "metal_hardware_manager: AGXAccelerator service not found "
                             "(non-Apple-Silicon environment?), GPU properties will use defaults\n";
    return IO_OBJECT_NULL;
  }

  // Get the GPU's entry object.
  io_iterator_t entry_iterator;
  kern_return_t error = IOServiceGetMatchingServices(
    kIOMainPortDefault, match_dictionary, &entry_iterator);
  if (error != kIOReturnSuccess) {
    HIPSYCL_DEBUG_WARNING << "metal_hardware_manager: No objects match AGXAccelerator service, "
                             "GPU properties will use defaults\n";
    return IO_OBJECT_NULL;
  }
  io_registry_entry_t gpu_entry = IOIteratorNext(entry_iterator);
  if (IOIteratorNext(entry_iterator)) {
    HIPSYCL_DEBUG_WARNING << "metal_hardware_manager: Found multiple GPUs\n";
  }

  // Release acquired objects.
  IOObjectRelease(entry_iterator);
  return gpu_entry;
}

// Number of GPU cores.
inline int64_t get_gpu_core_count(io_registry_entry_t gpu_entry) {
  if (gpu_entry == IO_OBJECT_NULL)
    return 1;
#if TARGET_OS_IPHONE
  // TODO: Determine the core count on iOS through something like DeviceKit.
  return 1;
#else
  // Get the number of cores.
  CFNumberRef gpu_core_count = (CFNumberRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
  if (!gpu_core_count) {
    HIPSYCL_DEBUG_WARNING << "get_gpu_core_count: Could not find 'gpu-core-count' property\n";
    return 1;
  }
  CFNumberType type = CFNumberGetType(gpu_core_count);
  if (type != kCFNumberSInt64Type) {
    HIPSYCL_DEBUG_WARNING << "get_gpu_core_count: 'gpu-core-count' not type sInt64\n";
    CFRelease(gpu_core_count);
    return 1;
  }
  int64_t value = 1;
  if (!CFNumberGetValue(gpu_core_count, type, &value)) {
    HIPSYCL_DEBUG_WARNING << "get_gpu_core_count: Could not fetch 'gpu-core-count' value\n";
  }

  // Release acquired objects.
  CFRelease(gpu_core_count);
  return value;
#endif
}

// Clock speed in MHz.
inline int64_t get_gpu_max_clock_speed(io_registry_entry_t gpu_entry) {
  if (gpu_entry == IO_OBJECT_NULL)
    return 0;
  CFStringRef model = (CFStringRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("model"), kCFAllocatorDefault, 0);
  if (!model) {
    HIPSYCL_DEBUG_WARNING << "get_gpu_max_clock_speed: Could not find 'model' property\n";
    return 0;
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
  if (gpu_entry == IO_OBJECT_NULL)
    return 0;
  CFStringRef model = (CFStringRef)IORegistryEntrySearchCFProperty(
    gpu_entry, kIOServicePlane, CFSTR("model"), kCFAllocatorDefault, 0);
  if (!model) {
    HIPSYCL_DEBUG_WARNING << "get_gpu_slc_size: Could not find 'model' property\n";
    return 0;
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
  // Try to get physical RAM size via sysctl.
  int64_t system_memory = 0;
  size_t size = sizeof(system_memory);
  int error = sysctlbyname("hw.memsize", &system_memory, &size, NULL, 0);
  if (error) {
    HIPSYCL_DEBUG_WARNING << "get_max_allocated_size: Could not query 'hw.memsize', "
                             "falling back to Metal reported limit\n";
  }

#if TARGET_OS_IPHONE
  if (system_memory > 0) {
    // The tested limit is ~3725 / 5494 MB on iOS, 67% of physical RAM.
    // We go slightly under (65%) for safety.
    int64_t working_set = (65 * (system_memory / 1024 / 1024) / 100) * 1024 * 1024;
    return working_set;
  }
  // Fallback: use Metal's recommended limit.
  int64_t metal_limit = static_cast<int64_t>(device->recommendedMaxWorkingSetSize());
  return metal_limit > 0 ? metal_limit : 256LL * 1024 * 1024;
#else
  int64_t metal_limit = static_cast<int64_t>(device->recommendedMaxWorkingSetSize());
  if (system_memory > 0 && metal_limit > 0) {
    // ~21700 / 32768 MB on macOS, 66% of physical RAM. We use 65% for safety.
    int64_t working_set = (65 * (system_memory / 1024 / 1024) / 100) * 1024 * 1024;
    return std::min(working_set, metal_limit);
  }
  if (metal_limit > 0)
    return metal_limit;
  if (system_memory > 0)
    return (65 * (system_memory / 1024 / 1024) / 100) * 1024 * 1024;
  // Last resort fallback (256 MB).
  return 256LL * 1024 * 1024;
#endif
}

} // namespace

metal_hardware_context::metal_hardware_context(MTL::Device* device)
    : _device{device}
{
  if (!_device) {
    throw std::runtime_error{"No Metal device found."};
  }

  if (device->supportsFamily(MTL::GPUFamily(MTL::GPUFamilyApple8 + 1))) {
    _gpu_family = std::numeric_limits<int>::max();
  } else if (device->supportsFamily(MTL::GPUFamilyApple8)) {
    _gpu_family = MTL::GPUFamilyApple8;
  } else if (device->supportsFamily(MTL::GPUFamilyApple7)) {
    _gpu_family = MTL::GPUFamilyApple7;
  }

  io_registry_entry_t gpu_entry = get_gpu_entry();

  _core_count = get_gpu_core_count(gpu_entry);
  _max_clock_speed = get_gpu_max_clock_speed(gpu_entry);
  _slc_size = get_gpu_slc_size(gpu_entry);
  _max_allocated_size = get_max_allocated_size(_device);

  if (gpu_entry != IO_OBJECT_NULL)
    IOObjectRelease(gpu_entry);
}

// metal_hardware_context
bool metal_hardware_context::is_cpu() const {
  return false;
}

bool metal_hardware_context::is_gpu() const {
  return true;
}

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

std::size_t metal_hardware_context::get_max_memcpy_concurrency() const {
  return 1; // TODO
}

std::string metal_hardware_context::get_device_name() const {
  return _device->name()->utf8String();
}

std::string metal_hardware_context::get_vendor_name() const {
  return "Apple";
}

std::string metal_hardware_context::get_device_arch() const {
  return _device->architecture()->name()->utf8String();
}

bool metal_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::images:
    return true;
  case device_support_aspect::error_correction:
    return false;
  case device_support_aspect::host_unified_memory:
    return _device->hasUnifiedMemory();
  case device_support_aspect::little_endian:
    return true;

  case device_support_aspect::global_mem_cache:
  case device_support_aspect::global_mem_cache_read_only:
  case device_support_aspect::global_mem_cache_read_write:
    return true;

  case device_support_aspect::emulated_local_memory:
    return false;
  case device_support_aspect::sub_group_independent_forward_progress:
    return false;

  case device_support_aspect::usm_device_allocations:
  case device_support_aspect::usm_host_allocations:
  case device_support_aspect::usm_shared_allocations:
    return true;

  case device_support_aspect::usm_atomic_host_allocations:
  case device_support_aspect::usm_atomic_shared_allocations:
  case device_support_aspect::usm_system_allocations:
    return false;

  case device_support_aspect::execution_timestamps:
    return false; // TODO: implement

  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return true;
#else
    return false;
#endif
  case device_support_aspect::work_item_independent_forward_progress:
    return false;

  case device_support_aspect::fp64:
    return false;
  case device_support_aspect::atomic64:
    return false;
  default:
    return false;
  }

  return false;
}

std::size_t metal_hardware_context::get_property(device_uint_property prop) const {
  using P = device_uint_property;

  switch(prop) {
  case P::max_compute_units:
    return _core_count;

  case P::max_work_group_range0:
  case P::max_work_group_range1:
  case P::max_work_group_range2:
    return 65535;

  case P::max_work_group_range_size:
    return 65535ull * 65535ull;

  case P::max_global_size0:
  case P::max_global_size1:
  case P::max_global_size2:
    return 65535;

  case P::needs_dimension_flip:
    return 0;

  case P::max_group_size0:
  case P::max_group_size1:
  case P::max_group_size2:
    return 1024;

  case P::max_group_size:
    return 1024;

  case P::max_num_sub_groups:
    return 32;

  case P::preferred_vector_width_char:
  case P::preferred_vector_width_short:
  case P::preferred_vector_width_int:
  case P::preferred_vector_width_long:
  case P::preferred_vector_width_half:
  case P::preferred_vector_width_float:
  case P::preferred_vector_width_double:
    return 1;

  case P::native_vector_width_char:
  case P::native_vector_width_short:
  case P::native_vector_width_int:
  case P::native_vector_width_long:
  case P::native_vector_width_half:
  case P::native_vector_width_float:
  case P::native_vector_width_double:
    return 1;

  case P::max_clock_speed:
    return _max_clock_speed;

  case P::max_malloc_size:
    return static_cast<std::size_t>(_device->maxBufferLength());

  case P::address_bits:
    return 64;

  case P::max_read_image_args:
  case P::max_write_image_args:
    return 128;

  case P::image2d_max_width:
  case P::image2d_max_height:
    return 16384;

  case P::image3d_max_width:
  case P::image3d_max_height:
  case P::image3d_max_depth:
    return 2048;

  case P::image_max_buffer_size:
    return static_cast<std::size_t>(_device->maxBufferLength());

  case P::image_max_array_size:
    return 2048;

  case P::max_samplers:
    return 16;

  case P::max_parameter_size:
    return 4096;

  case P::mem_base_addr_align:
    return 4096;

  case P::global_mem_cache_line_size:
    return 128;

  case P::global_mem_cache_size:
    return _slc_size;

  case P::global_mem_size:
    return _max_allocated_size;

  case P::max_constant_buffer_size:
    return 4096;

  case P::max_constant_args:
    return 16;

  case P::local_mem_size:
    return 32 * 1024;

  case P::printf_buffer_size:
    return 0;

  case P::partition_max_sub_devices:
    return 0;

  case P::vendor_id:
    return 0x1027f00;

  case P::architecture:
    return static_cast<std::size_t>(_device->registryID());

  case P::backend_id:
    return static_cast<std::size_t>(backend_id::metal);

  case P::queue_priority_range_low:
    return 0;
  case P::queue_priority_range_high:
    return 0;

  default:
    return 0;
  }
}

std::vector<std::size_t>
metal_hardware_context::get_property(device_uint_list_property prop) const {
  using P = device_uint_list_property;

  switch(prop) {
  case P::sub_group_sizes:
    return {32};
  }

  return {};
}

std::string metal_hardware_context::get_driver_version() const {
  std::string version;
  version = NS::ProcessInfo::processInfo()->operatingSystemVersionString()->utf8String();

  struct feature_set_name {
    uint64_t value;
    const char* name;
  };
  static const feature_set_name known_features[] = {
    {0, "iOS_GPUFamily1_v1"},
    {1, "iOS_GPUFamily2_v1"},
    {2, "iOS_GPUFamily1_v2"},
    {3, "iOS_GPUFamily2_v2"},
    {4, "iOS_GPUFamily3_v1"},
    {5, "iOS_GPUFamily1_v3"},
    {6, "iOS_GPUFamily2_v3"},
    {7, "iOS_GPUFamily3_v2"},
    {8, "iOS_GPUFamily1_v4"},
    {9, "iOS_GPUFamily2_v4"},
    {10, "iOS_GPUFamily3_v3"},
    {11, "iOS_GPUFamily4_v1"},
    {12, "iOS_GPUFamily1_v5"},
    {13, "iOS_GPUFamily2_v5"},
    {14, "iOS_GPUFamily3_v4"},
    {15, "iOS_GPUFamily4_v2"},
    {16, "iOS_GPUFamily5_v1"},
    {10000, "macOS_GPUFamily1_v1"},
    {10001, "macOS_GPUFamily1_v2"},
    {10002, "macOS_ReadWriteTextureTier2"},
    {10003, "macOS_GPUFamily1_v3"},
    {10004, "macOS_GPUFamily1_v4"},
    {10005, "macOS_GPUFamily2_v1"},
    {20000, "watchOS_GPUFamily1_v1"},
    {20001, "watchOS_GPUFamily2_v1"},
    {30000, "tvOS_GPUFamily1_v1"},
    {30001, "tvOS_GPUFamily1_v2"},
    {30002, "tvOS_GPUFamily1_v3"},
    {30003, "tvOS_GPUFamily2_v1"},
    {30004, "tvOS_GPUFamily1_v4"},
    {30005, "tvOS_GPUFamily2_v2"}
  };

  std::vector<std::string> features;
  for (uint64_t fs = 0; fs <= 16; ++fs) {
    if (_device->supportsFeatureSet(static_cast<MTL::FeatureSet>(fs))) {
      const char* name = nullptr;
      for (const auto& f : known_features) {
        if (f.value == fs) { name = f.name; break; }
      }
      if (name)
        features.emplace_back(name);
      else
        features.emplace_back("Feature" + std::to_string(fs));
    }
  }
  for (uint64_t fs = 10000; fs <= 10005; ++fs) {
    if (_device->supportsFeatureSet(static_cast<MTL::FeatureSet>(fs))) {
      const char* name = nullptr;
      for (const auto& f : known_features) {
        if (f.value == fs) { name = f.name; break; }
      }
      if (name)
        features.emplace_back(name);
      else
        features.emplace_back("Feature" + std::to_string(fs));
    }
  }
  for (uint64_t fs = 20000; fs <= 20001; ++fs) {
    if (_device->supportsFeatureSet(static_cast<MTL::FeatureSet>(fs))) {
      const char* name = nullptr;
      for (const auto& f : known_features) {
        if (f.value == fs) { name = f.name; break; }
      }
      if (name)
        features.emplace_back(name);
      else
        features.emplace_back("Feature" + std::to_string(fs));
    }
  }
  for (uint64_t fs = 30000; fs <= 30005; ++fs) {
    if (_device->supportsFeatureSet(static_cast<MTL::FeatureSet>(fs))) {
      const char* name = nullptr;
      for (const auto& f : known_features) {
        if (f.value == fs) { name = f.name; break; }
      }
      if (name)
        features.emplace_back(name);
      else
        features.emplace_back("Feature" + std::to_string(fs));
    }
  }

  std::string features_str;
  for (size_t i = 0; i < features.size(); ++i) {
    features_str += features[i];
    if (i + 1 < features.size())
      features_str += ",";
  }

  // GPUFamily support
  struct gpu_family_name {
    int64_t value;
    const char* name;
  };
  static const gpu_family_name known_families[] = {
    {1001, "Apple1"}, {1002, "Apple2"}, {1003, "Apple3"}, {1004, "Apple4"}, {1005, "Apple5"},
    {1006, "Apple6"}, {1007, "Apple7"}, {1008, "Apple8"}, {1009, "Apple9"}, {1010, "Apple10"},
    {2001, "Mac1"}, {2002, "Mac2"},
    {3001, "Common1"}, {3002, "Common2"}, {3003, "Common3"},
    {4001, "MacCatalyst1"}, {4002, "MacCatalyst2"},
    {5001, "Metal3"}, {5002, "Metal4"}
  };
  std::vector<std::string> families;
  for (int64_t fam = 1001; fam <= 1010; ++fam) {
    if (_device->supportsFamily(static_cast<MTL::GPUFamily>(fam))) {
      const char* name = nullptr;
      for (const auto& f : known_families) { if (f.value == fam) { name = f.name; break; } }
      if (name) families.emplace_back(name); else families.emplace_back("GPUFamily" + std::to_string(fam));
    }
  }
  for (int64_t fam = 2001; fam <= 2002; ++fam) {
    if (_device->supportsFamily(static_cast<MTL::GPUFamily>(fam))) {
      const char* name = nullptr;
      for (const auto& f : known_families) { if (f.value == fam) { name = f.name; break; } }
      if (name) families.emplace_back(name); else families.emplace_back("GPUFamily" + std::to_string(fam));
    }
  }
  for (int64_t fam = 3001; fam <= 3003; ++fam) {
    if (_device->supportsFamily(static_cast<MTL::GPUFamily>(fam))) {
      const char* name = nullptr;
      for (const auto& f : known_families) { if (f.value == fam) { name = f.name; break; } }
      if (name) families.emplace_back(name); else families.emplace_back("GPUFamily" + std::to_string(fam));
    }
  }
  for (int64_t fam = 4001; fam <= 4002; ++fam) {
    if (_device->supportsFamily(static_cast<MTL::GPUFamily>(fam))) {
      const char* name = nullptr;
      for (const auto& f : known_families) { if (f.value == fam) { name = f.name; break; } }
      if (name) families.emplace_back(name); else families.emplace_back("GPUFamily" + std::to_string(fam));
    }
  }
  for (int64_t fam = 5001; fam <= 5002; ++fam) {
    if (_device->supportsFamily(static_cast<MTL::GPUFamily>(fam))) {
      const char* name = nullptr;
      for (const auto& f : known_families) { if (f.value == fam) { name = f.name; break; } }
      if (name) families.emplace_back(name); else families.emplace_back("GPUFamily" + std::to_string(fam));
    }
  }
  std::string families_str;
  for (size_t i = 0; i < families.size(); ++i) {
    families_str += families[i];
    if (i + 1 < families.size())
      families_str += ",";
  }

  // ArgumentBuffersTier
  std::string argbuf_tier;
  switch (_device->argumentBuffersSupport()) {
    case MTL::ArgumentBuffersTier1: argbuf_tier = "ArgumentBuffersTier1"; break;
    case MTL::ArgumentBuffersTier2: argbuf_tier = "ArgumentBuffersTier2"; break;
    default: argbuf_tier = "Unknown"; break;
  }

  std::string result = version;
  if (!features_str.empty()) result += ", featuresets: " + features_str;
  if (!families_str.empty()) result += ", gpufamilies: " + families_str;
  result += ", argumentBuffersTier: " + argbuf_tier;
  return result;
}

std::string metal_hardware_context::get_profile() const {
  return "FULL_PROFILE";
}

std::size_t metal_hardware_context::get_platform_index() const {
  return 0;
}

metal_hardware_context::~metal_hardware_context() = default;

// metal_hardware_manager

metal_hardware_manager::metal_hardware_manager()
{
  auto device = MTL::CreateSystemDefaultDevice();
  if (device) {
    auto id = device_id{
      backend_descriptor{hardware_platform::metal, api_platform::metal},
      0
    };
    _devices.emplace_back(device);
    _contexts.emplace_back(metal_hardware_context{device});
    _allocators.emplace_back(device, id);
  }
}

metal_inorder_queue* metal_hardware_manager::make_queue(size_t index) {
  return new metal_inorder_queue(
    _devices[index],
    &_allocators[index],
    device_id{
      backend_descriptor{hardware_platform::metal, api_platform::metal},
      static_cast<int>(index)
    });
}

std::size_t metal_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *metal_hardware_manager::get_device(std::size_t index) {
  return &_contexts[index];
}

device_id metal_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{
    backend_descriptor{hardware_platform::metal, api_platform::metal},
    static_cast<int>(index)
  };
}

std::size_t metal_hardware_manager::get_num_platforms() const {
  return 1;
}

metal_allocator* metal_hardware_manager::get_allocator(size_t index) {
  return &_allocators[index];
}

metal_hardware_manager::~metal_hardware_manager() {
  for (auto device : _devices) {
    if (device) {
      device->release();
    }
  }
}

} // namespace rt
} // namespace hipsycl