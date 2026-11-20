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

#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"
#include "hipSYCL/common/config.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/vk/vk_allocator.hpp"
#include <algorithm>
#include <limits>
#include <sstream>

namespace hipsycl {
namespace rt {

vk_hardware_context::vk_hardware_context(
    const vk::raii::PhysicalDevice &phys_device, int dev_id, uint16_t features)
    : _physical_device(phys_device), _dev_id(dev_id),
      _physical_dev_features(features), _enabled_extensions(0) {
  _properties = phys_device.getProperties();

  std::vector<vk::QueueFamilyProperties> queue_family_props =
      phys_device.getQueueFamilyProperties();

  // Queue family that supports compute and the least other features
  vk::Flags<vk::QueueFlagBits> chosen_queue_flags{};
  for (uint32_t i = 0; i < queue_family_props.size(); i++) {
    const auto flags = queue_family_props[i].queueFlags;
    if (flags & vk::QueueFlagBits::eCompute) {
      if (_queue_index == UINT32_MAX || flags < chosen_queue_flags) {
        _queue_index = i;
        chosen_queue_flags = flags;
      }
    }
  }
  if (_queue_index == UINT32_MAX) {
    print_error(__acpp_here(),
                error_info{std::string(
                    "vk_hardware_context: could not find a compute queue")});
  }

  float queue_priority = 1.0f;
  vk::DeviceQueueCreateInfo queue_create_info{
      {}, _queue_index, 1, &queue_priority};

  vk::PhysicalDeviceVulkan12Features phys_dev_12_features{};
  phys_dev_12_features.bufferDeviceAddress = VK_TRUE;
  phys_dev_12_features.timelineSemaphore = VK_TRUE;
  phys_dev_12_features.shaderSubgroupExtendedTypes = VK_TRUE;

  phys_dev_12_features.shaderInt8 =
      (_physical_dev_features & vk_device_features::shaderInt8) ? VK_TRUE
                                                                : VK_FALSE;
  phys_dev_12_features.shaderFloat16 =
      (_physical_dev_features & vk_device_features::shaderFloat16) ? VK_TRUE
                                                                   : VK_FALSE;
  phys_dev_12_features.storagePushConstant8 =
      (_physical_dev_features & vk_device_features::storagePushConstant8)
          ? VK_TRUE
          : VK_FALSE;
  phys_dev_12_features.uniformAndStorageBuffer8BitAccess =
      (_physical_dev_features &
       vk_device_features::uniformAndStorageBuffer8BitAccess)
          ? VK_TRUE
          : VK_FALSE;

  vk::PhysicalDeviceVulkan11Features phys_dev_11_features{};
  phys_dev_11_features.variablePointers =
      (_physical_dev_features & vk_device_features::variablePointers)
          ? VK_TRUE
          : VK_FALSE;
  phys_dev_11_features.variablePointersStorageBuffer =
      (_physical_dev_features &
       vk_device_features::variablePointersStorageBuffer)
          ? VK_TRUE
          : VK_FALSE;
  phys_dev_11_features.storagePushConstant16 =
      (_physical_dev_features & vk_device_features::storagePushConstant16)
          ? VK_TRUE
          : VK_FALSE;
  phys_dev_11_features.uniformAndStorageBuffer16BitAccess =
      (_physical_dev_features &
       vk_device_features::uniformAndStorageBuffer16BitAccess)
          ? VK_TRUE
          : VK_FALSE;

  vk::PhysicalDeviceFeatures phys_dev_features{};
  phys_dev_features.shaderInt16 =
      (_physical_dev_features & vk_device_features::shaderInt16) ? VK_TRUE
                                                                 : VK_FALSE;
  phys_dev_features.shaderInt64 =
      (_physical_dev_features & vk_device_features::shaderInt64) ? VK_TRUE
                                                                 : VK_FALSE;
  phys_dev_features.shaderFloat64 =
      (_physical_dev_features & vk_device_features::shaderFloat64) ? VK_TRUE
                                                                   : VK_FALSE;

  vk::DeviceCreateInfo dev_create_info{{}, queue_create_info};

  auto phys_dev_extensions =
      _physical_device.enumerateDeviceExtensionProperties();

  // Portability subset extensions used to enable MoltenVK.
  // Calibrated timestamps extension used to enable event profiling.
  // Shader core properties/builtins extensions used to query
  // max_compute_units, not EXT/KHR so we need the vendor extensions.
  std::array<std::pair<const char *, uint16_t>, 5> optional_exts = {
      std::make_pair(VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME,
                     vk_device_extensions::khr_portability_subset),
      std::make_pair(VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
                     vk_device_extensions::khr_calibrated_timestamps),
      std::make_pair(VK_EXT_CALIBRATED_TIMESTAMPS_EXTENSION_NAME,
                     vk_device_extensions::ext_calibrated_timestamps),
      std::make_pair(VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME,
                     vk_device_extensions::amd_shader_core_properties),
      std::make_pair(VK_ARM_SHADER_CORE_BUILTINS_EXTENSION_NAME,
                     vk_device_extensions::arm_shader_core_builtins)};
  std::vector<const char *> enable_ext_names;
  for (auto opt_ext : optional_exts) {
    bool ext_supported = std::any_of(
        std::begin(phys_dev_extensions), std::end(phys_dev_extensions),
        [opt_ext](auto const &ext_property) {
          return strcmp(ext_property.extensionName, opt_ext.first) == 0;
        });
    if (ext_supported) {
      HIPSYCL_DEBUG_INFO << "vk_hardware_context: enabling extension "
                         << opt_ext.first << std::endl;
      _enabled_extensions |= opt_ext.second;
      enable_ext_names.push_back(opt_ext.first);
    }
  }

  if (auto num_dev_extensions = enable_ext_names.size();
      num_dev_extensions != 0) {
    dev_create_info.setEnabledExtensionCount(num_dev_extensions);
    dev_create_info.setPpEnabledExtensionNames(enable_ext_names.data());
  }

  vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan12Features,
                     vk::PhysicalDeviceVulkan11Features>
      dev_create_info_chain(dev_create_info, phys_dev_features,
                            phys_dev_12_features, phys_dev_11_features);
  auto device_create_info = dev_create_info_chain.get<vk::DeviceCreateInfo>();

  _device = vk::raii::Device(phys_device, device_create_info);
  _queue = vk::raii::Queue(_device, _queue_index, 0);

  _limits = phys_device.getProperties().limits;

  /* Vulkan hpp headers don't seem to like instantiating
   * a pointer chain wrapper with `vkGetPhysicalDeviceSubgroupProperties`
   * Workaround by using C API to query this.
   */

  VkPhysicalDeviceSubgroupSizeControlProperties sg_control_props{};
  sg_control_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;

  VkPhysicalDeviceShaderCorePropertiesAMD amd_shader_core_props{};
  amd_shader_core_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_PROPERTIES_AMD;

  VkPhysicalDeviceShaderCoreBuiltinsPropertiesARM arm_shader_core_props{};
  arm_shader_core_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CORE_BUILTINS_PROPERTIES_ARM;

  if (are_extensions_enabled(
          vk_device_extensions::amd_shader_core_properties)) {
    sg_control_props.pNext = &amd_shader_core_props;
  } else if (are_extensions_enabled(
                 vk_device_extensions::arm_shader_core_builtins)) {
    sg_control_props.pNext = &arm_shader_core_props;
  }

  VkPhysicalDeviceSubgroupProperties sg_props{};
  sg_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  sg_props.pNext = &sg_control_props;

  VkPhysicalDeviceMaintenance3Properties maint3_props;
  maint3_props.sType =
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
  maint3_props.pNext = &sg_props;

  VkPhysicalDeviceProperties2KHR properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
  properties.pNext = &maint3_props;

  vkGetPhysicalDeviceProperties2(*phys_device, &properties);
  _subgroup_size = sg_props.subgroupSize;

  if (are_extensions_enabled(
          vk_device_extensions::amd_shader_core_properties)) {
    // AMD GPUs are organized hierarchically into shader engines, each
    // containing a number of shader arrays, each of which in turn contains a
    // number of compute units.
    _max_compute_units = amd_shader_core_props.shaderEngineCount *
                         amd_shader_core_props.shaderArraysPerEngineCount *
                         amd_shader_core_props.computeUnitsPerShaderArray;
  } else if (are_extensions_enabled(
                 vk_device_extensions::arm_shader_core_builtins)) {
    _max_compute_units = arm_shader_core_props.shaderCoreCount;
  }

  if (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) {
    _physical_dev_features |= vk_device_features::groupNonUniform;
  }
  if (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_VOTE_BIT) {
    _physical_dev_features |= vk_device_features::groupNonUniformVote;
  }
  if (sg_props.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) {
    _physical_dev_features |= vk_device_features::groupNonUniformShuffle;
  }

  _max_num_subgroups = sg_control_props.maxComputeWorkgroupSubgroups;
  _max_alloc_size = maint3_props.maxMemoryAllocationSize;

  HIPSYCL_DEBUG_INFO << "vk_hardware_context: logical device constructed "
                     << "for device id " << dev_id << " with queue family "
                     << "index " << _queue_index << std::endl;
}

void vk_hardware_context::init() {
  device_id dev{backend_descriptor{hardware_platform::vk, api_platform::vk},
                _dev_id};
  _allocator = std::make_unique<vk_allocator>(this, dev);
}

bool vk_hardware_context::is_cpu() const {
  return vk::PhysicalDeviceType::eCpu == _properties.deviceType;
}

bool vk_hardware_context::is_gpu() const {
  switch (_properties.deviceType) {
  case vk::PhysicalDeviceType::eIntegratedGpu:
  case vk::PhysicalDeviceType::eDiscreteGpu:
  case vk::PhysicalDeviceType::eVirtualGpu:
    return true;
  default:
    return false;
  }
}

std::size_t vk_hardware_context::get_max_kernel_concurrency() const {
  return 1;
}
std::size_t vk_hardware_context::get_max_memcpy_concurrency() const {
  return 1;
}

std::string vk_hardware_context::get_device_name() const {
  return _properties.deviceName;
}

std::string vk_hardware_context::get_vendor_name() const {
  std::stringstream stream;
  stream << "Vulkan - 0x" << std::hex << _properties.vendorID;
  return stream.str();
}

std::string vk_hardware_context::get_device_arch() const { return "spirv"; }

bool vk_hardware_context::has(device_support_aspect aspect) const {
  switch (aspect) {
  case device_support_aspect::emulated_local_memory:
    return false;
  case device_support_aspect::host_unified_memory:
    switch (_properties.deviceType) {
    case vk::PhysicalDeviceType::eCpu:
      return true;
    default:
      return false;
    }
  case device_support_aspect::error_correction:
    return false;
  case device_support_aspect::global_mem_cache:
    return false;
  case device_support_aspect::global_mem_cache_read_only:
    return false;
  case device_support_aspect::global_mem_cache_read_write:
    return false;
  case device_support_aspect::images:
    return false;
  case device_support_aspect::little_endian:
    return true;
  case device_support_aspect::sub_group_independent_forward_progress:
    return false;
  case device_support_aspect::usm_device_allocations:
    return true;
  case device_support_aspect::usm_host_allocations:
    return false;
  case device_support_aspect::usm_atomic_host_allocations:
    return false;
  case device_support_aspect::usm_shared_allocations:
    return false;
  case device_support_aspect::usm_atomic_shared_allocations:
    return false;
  case device_support_aspect::usm_system_allocations:
    return false;
  case device_support_aspect::execution_timestamps: {
    // Verify that device can create a query pool on the compute queue so
    // enable capturing asynchronous timestamps during command-buffer execution
    if (_limits.timestampPeriod == 0) {
      return false;
    }

    const vk::QueueFamilyProperties queue_family_props =
        _physical_device.getQueueFamilyProperties()[_queue_index];
    if (!queue_family_props.timestampValidBits) {
      return false;
    }

    // Verify that device has either KHR or EXT calibrated timestamps extension
    // that support both clock monotonic & device domains
    std::pair<bool, bool> required_domains(false, false);

    // vkTimeDomainKHR is aliased to vkTimeDomainEXT
    auto check_required_domains =
        [&required_domains](vk::TimeDomainEXT &time_domain) {
          if (time_domain == vk::TimeDomainEXT::eClockMonotonic) {
            required_domains.first = true;
          }

          if (time_domain == vk::TimeDomainEXT::eDevice) {
            required_domains.second = true;
          }
        };

    // Prefer using KHR entry point rather than EXT if a device supports both
    if (are_extensions_enabled(
            vk_device_extensions::khr_calibrated_timestamps)) {
      auto time_domains = _physical_device.getCalibrateableTimeDomainsKHR();
      std::for_each(std::begin(time_domains), std::end(time_domains),
                    check_required_domains);
    } else if (are_extensions_enabled(
                   vk_device_extensions::ext_calibrated_timestamps)) {
      auto time_domains = _physical_device.getCalibrateableTimeDomainsEXT();
      std::for_each(std::begin(time_domains), std::end(time_domains),
                    check_required_domains);
    }

    return required_domains.first && required_domains.second;
  }
  case device_support_aspect::sscp_kernels:
#ifdef HIPSYCL_WITH_SSCP_COMPILER
    return true;
#else
    return false;
#endif
  case device_support_aspect::work_item_independent_forward_progress:
    return false;
  case device_support_aspect::fp64:
    // Not supported by clspv
    return false;
  case device_support_aspect::atomic64:
    // Not supported by clspv
    // https://github.com/google/clspv/blob/main/docs/OpenCLCOnVulkan.md#opencl-20-atomic-functions
    return false;
  case device_support_aspect::free_memory:
    return false;
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t vk_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return _max_compute_units;
  case device_uint_property::max_work_group_range0:
    return _limits.maxComputeWorkGroupCount[0];
  case device_uint_property::max_work_group_range1:
    return _limits.maxComputeWorkGroupCount[1];
  case device_uint_property::max_work_group_range2:
    return _limits.maxComputeWorkGroupCount[2];
  case device_uint_property::max_work_group_range_size:
    return std::numeric_limits<std::size_t>::max();
  case device_uint_property::max_global_size0:
    return _limits.maxComputeWorkGroupSize[0] *
           _limits.maxComputeWorkGroupCount[0];
  case device_uint_property::max_global_size1:
    return _limits.maxComputeWorkGroupSize[1] *
           _limits.maxComputeWorkGroupCount[1];
  case device_uint_property::max_global_size2:
    return _limits.maxComputeWorkGroupSize[2] *
           _limits.maxComputeWorkGroupCount[2];
  case device_uint_property::max_group_size0:
    return _limits.maxComputeWorkGroupSize[0];
  case device_uint_property::max_group_size1:
    return _limits.maxComputeWorkGroupSize[1];
  case device_uint_property::max_group_size2:
    return _limits.maxComputeWorkGroupSize[2];
  case device_uint_property::max_group_size:
    return _limits.maxComputeWorkGroupInvocations;
  case device_uint_property::max_num_sub_groups:
    return _max_num_subgroups;
  case device_uint_property::needs_dimension_flip:
    return true;
  // Vulkan/SPIR-V has no notion of a preferred/native vector width. We
  // therefore report a width of 1 (no vectorization advantage) for every
  // scalar type the device supports in shader code, and 0 for types that
  // require an unsupported capability.
  case device_uint_property::preferred_vector_width_char:
  case device_uint_property::native_vector_width_char:
    return (_physical_dev_features & vk_device_features::shaderInt8) ? 1 : 0;
  case device_uint_property::preferred_vector_width_short:
  case device_uint_property::native_vector_width_short:
    return (_physical_dev_features & vk_device_features::shaderInt16) ? 1 : 0;
  case device_uint_property::preferred_vector_width_int:
  case device_uint_property::native_vector_width_int:
    // 32-bit integers are always supported by Vulkan compute shaders.
    return 1;
  case device_uint_property::preferred_vector_width_long:
  case device_uint_property::native_vector_width_long:
    return (_physical_dev_features & vk_device_features::shaderInt64) ? 1 : 0;
  case device_uint_property::preferred_vector_width_half:
  case device_uint_property::native_vector_width_half:
    return (_physical_dev_features & vk_device_features::shaderFloat16) ? 1 : 0;
  case device_uint_property::preferred_vector_width_float:
  case device_uint_property::native_vector_width_float:
    // 32-bit floats are always supported by Vulkan compute shaders.
    return 1;
  case device_uint_property::preferred_vector_width_double:
  case device_uint_property::native_vector_width_double:
    return (_physical_dev_features & vk_device_features::shaderFloat64) ? 1 : 0;
  case device_uint_property::max_clock_speed:
    // Vulkan has no API or extensions for querying the device clock frequency.
    return 0;
  case device_uint_property::max_malloc_size:
    return std::min(_allocator->get_global_mem_size(), _max_alloc_size);
  case device_uint_property::address_bits:
    return 64; // spirv64 only for now
  case device_uint_property::max_read_image_args:
  case device_uint_property::max_write_image_args:
  case device_uint_property::image2d_max_width:
  case device_uint_property::image2d_max_height:
  case device_uint_property::image3d_max_width:
  case device_uint_property::image3d_max_height:
  case device_uint_property::image3d_max_depth:
  case device_uint_property::image_max_buffer_size:
  case device_uint_property::image_max_array_size:
  case device_uint_property::max_samplers:
    return 0; // Images are not currently supported
  case device_uint_property::max_parameter_size:
    // Scalar/POD kernel arguments are passed via push constants, but
    // arguments that do not fit fallback to being lowered to uniform-buffers.
    // So there is no single hard limit, but report the push constant limit,
    // which bounds the "fast path" used for ordinary/small argument lists.
    return _limits.maxPushConstantsSize;
  case device_uint_property::mem_base_addr_align: {
    // Reports the bit-width of the largest built-in SYCL vector type this
    // device can represent (`sycl::vec<T,16>`)
    const bool has_8_byte_scalar_type =
        (_physical_dev_features & vk_device_features::shaderInt64) ||
        (_physical_dev_features & vk_device_features::shaderFloat64);
    const std::size_t largest_scalar_bytes = has_8_byte_scalar_type ? 8 : 4;
    return 16 * largest_scalar_bytes * 8;
  }
  case device_uint_property::global_mem_cache_line_size:
  case device_uint_property::global_mem_cache_size:
    // Vulkan has no portable query for cache sizes, and has() already
    // reports global_mem_cache/_read_only/_read_write as unsupported.
    return 0;
  case device_uint_property::global_mem_size:
    return _allocator->get_global_mem_size();
  case device_uint_property::max_constant_buffer_size:
    // Constant kernel arguments are passed as uniform buffer bindings
    return _limits.maxUniformBufferRange;
  case device_uint_property::max_constant_args:
    // Every constant argument that shares a binding is merged into one
    // descriptor, addressed by offset within a single buffer.
    // clspv can pack arbitrarily many of them into few/one binding(s). The
    // real limit is the total byte size of such a binding,
    // reported via max_constant_buffer_size, so there is no separate
    // hard cap on argument count to report here.
    return std::numeric_limits<std::size_t>::max();
  case device_uint_property::local_mem_size:
    return _limits.maxComputeSharedMemorySize;
  case device_uint_property::printf_buffer_size:
    return std::numeric_limits<std::size_t>::max();
  case device_uint_property::partition_max_sub_devices:
    return 0;
  case device_uint_property::vendor_id:
    return _properties.vendorID;
  case device_uint_property::architecture:
    // Vulkan deviceID is a unique identifier for the physical device among
    // devices available from the vendor
    return _properties.deviceID;
  case device_uint_property::backend_id:
    return static_cast<int>(backend_id::vk);
  case device_uint_property::queue_priority_range_low:
  case device_uint_property::queue_priority_range_high:
    return 0;
  case device_uint_property::free_memory:
    return 0;
  }
  assert(false && "Invalid device property");
  std::terminate();
}

std::vector<std::size_t>
vk_hardware_context::get_property(device_uint_list_property prop) const {
  switch (prop) {
  case device_uint_list_property::sub_group_sizes:
    return std::vector<std::size_t>{_subgroup_size};
  }

  assert(false && "Invalid device property");
  std::terminate();
}

std::string vk_hardware_context::get_driver_version() const {
  auto properties2 =
      _physical_device.getProperties2<vk::PhysicalDeviceProperties2,
                                      vk::PhysicalDeviceDriverProperties>();
  vk::PhysicalDeviceDriverProperties const &driver_props =
      properties2.get<vk::PhysicalDeviceDriverProperties>();
  std::string driver(std::string(driver_props.driverName));
  driver.append(", ");
  driver.append(std::string(driver_props.driverInfo));
  return driver;
}

std::string vk_hardware_context::get_profile() const { return "FULL_PROFILE"; }

std::size_t vk_hardware_context::get_platform_index() const { return 0; }

vk_allocator *vk_hardware_context::get_allocator() { return _allocator.get(); }

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
  std::string callback_msg(pCallbackData->pMessage);
  std::string kernel_name_str(
      "vkCreateComputePipelines(): pCreateInfos[0].stage.pName exceeds max "
      "length 256");
  if (callback_msg.find(kernel_name_str) != std::string::npos) {
    // Swallow validation error about max length of kernel name exceeding 256
    // chars. It's only the restriction about the string being null terminated
    // in that appears in the Vulkan spec, not the arbitrary 256 limit that the
    // validation layer has chosen
    return vk::False;
  }

  if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
      severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
    std::stringstream error_output;
    error_output << "vk_hardware_context validation layer: type"
                 << to_string(type) << " message: " << callback_msg;
    print_error(__acpp_here(), error_info{error_output.str()});
  }

  return vk::False;
}

const std::vector<const char *> vk_hardware_manager::_validation_layers = {
    "VK_LAYER_KHRONOS_validation"};

vk_hardware_manager::vk_hardware_manager()
    : _hw_platform{hardware_platform::vk} {
  constexpr vk::ApplicationInfo app_info{
      "AdaptiveCpp", // pApplicationName
      VK_MAKE_VERSION(ACPP_VERSION_MAJOR, ACPP_VERSION_MINOR,
                      ACPP_VERSION_PATCH), // applicationVersion
      "No Engine",                         // pEngineName
      VK_MAKE_VERSION(1, 0, 0),            // engineVersion
      vk::ApiVersion14                     // apiVersion
  };

  // Enable validation layer if requested
  std::vector<char const *> required_layers;
#ifndef NDEBUG
  required_layers.assign(_validation_layers.begin(), _validation_layers.end());
#endif

  auto layer_properties = _context.enumerateInstanceLayerProperties();
  for (auto const &required_layer : required_layers) {
    bool layer_unsupported = std::none_of(
        std::begin(layer_properties), std::end(layer_properties),
        [required_layer](auto const &layer_property) {
          return strcmp(layer_property.layerName, required_layer) == 0;
        });
    if (layer_unsupported) {
      print_error(
          __acpp_here(),
          error_info{std::string(
                         "vk_hardware_manager: Required layer not supported - ")
                         .append(std::string(required_layer))});
    }
  }

  std::vector<const char *> enabled_extensions;
  auto ext_properties = _context.enumerateInstanceExtensionProperties();
  auto required_ext = vk::EXTDebugUtilsExtensionName;
  bool ext_unsupported = std::none_of(
      std::begin(ext_properties), std::end(ext_properties),
      [required_ext](auto const &ext_property) {
        return strcmp(ext_property.extensionName, required_ext) == 0;
      });
  if (ext_unsupported) {
    print_error(
        __acpp_here(),
        error_info{
            std::string(
                "vk_hardware_manager: Required extension not supported - ")
                .append(std::string(required_ext))});
  }
  enabled_extensions.push_back(required_ext);

  // To layer ontop of moltenVK we need to opt-in to the Vulkan loader showing
  // non-conformant Vulkan implementations
  auto portability_ext = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
  bool portability_ext_supported = std::any_of(
      std::begin(ext_properties), std::end(ext_properties),
      [portability_ext](auto const &ext_property) {
        return strcmp(ext_property.extensionName, portability_ext) == 0;
      });

  vk::InstanceCreateFlags flags{};
  if (portability_ext_supported) {
    HIPSYCL_DEBUG_INFO << "vk_hardware_manager: enabling extension "
                       << portability_ext << std::endl;
    enabled_extensions.push_back(portability_ext);
    flags |= vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  }

  vk::InstanceCreateInfo create_info{
      flags,
      &app_info,                                        // pApplicationInfo
      static_cast<uint32_t>(required_layers.size()),    // enabledLayerCount
      required_layers.data(),                           // ppEnabledLayerNames
      static_cast<uint32_t>(enabled_extensions.size()), // enabledExtensionCount
      enabled_extensions.data() // ppEnabledExtensionNames
  };

#ifndef NDEBUG
  vk::DebugUtilsMessageSeverityFlagsEXT severity_flags(
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT message_type_flags(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  vk::DebugUtilsMessengerCreateInfoEXT debug_utils_messenger_create_info{
      {}, severity_flags, message_type_flags, &debugCallback};

  create_info.pNext = &debug_utils_messenger_create_info;
  _instance = vk::raii::Instance(_context, create_info);

  _debug_messenger =
      _instance.createDebugUtilsMessengerEXT(debug_utils_messenger_create_info);
#else
  _instance = vk::raii::Instance(_context, create_info);
#endif

  // Iterate over physical devices and find those capable being instantiated
  // into a logical device representing a SYCL device.
  const auto visibility_mask =
      application::get_settings().get<setting::visibility_mask>();
  std::vector<vk::raii::PhysicalDevice> devices =
      _instance.enumeratePhysicalDevices();
  int device_index = 0;
  for (const auto &phys_dev : devices) {
    // Check device supports at least v1.2
    bool supports_vulkan1_2 =
        phys_dev.getProperties().apiVersion >= VK_API_VERSION_1_2;
    if (!supports_vulkan1_2) {
      HIPSYCL_DEBUG_INFO << "vk_hardware_manager: physical device "
                         << device_index
                         << " doesn't support Vulkan 1.2, skipping."
                         << std::endl;
      device_index++;
      continue;
    }

    // Check device has at least 1 compute queue
    auto queue_families = phys_dev.getQueueFamilyProperties();
    bool supports_compute = std::any_of(
        queue_families.begin(), queue_families.end(), [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eCompute);
        });
    if (!supports_compute) {
      HIPSYCL_DEBUG_INFO << "vk_hardware_manager: physical device "
                         << device_index
                         << " doesn't support compute queue, skipping."
                         << std::endl;
      device_index++;
      continue;
    }

    auto supported_features =
        phys_dev.getFeatures2<vk::PhysicalDeviceFeatures2,
                              vk::PhysicalDeviceVulkan11Features,
                              vk::PhysicalDeviceVulkan12Features>();
    auto const &features_12 =
        supported_features.get<vk::PhysicalDeviceVulkan12Features>();
    // Essential for supporting USM, don't create a backend device without it
    if (!features_12.bufferDeviceAddress) {
      HIPSYCL_DEBUG_INFO << "vk_hardware_manager: physical device "
                         << device_index
                         << " doesn't support bufferDeviceAddress, skipping."
                         << std::endl;
      device_index++;
      continue;
    }
    // Essential for synchronization, don't create a backend device without it.
    if (!features_12.timelineSemaphore) {
      HIPSYCL_DEBUG_INFO << "vk_hardware_manager: physical device "
                         << device_index
                         << " doesn't support timeline semaphore, skipping."
                         << std::endl;
      device_index++;
      continue;
    }
    if (!features_12.shaderSubgroupExtendedTypes) {
      HIPSYCL_DEBUG_INFO
          << "vk_hardware_manager: physical device " << device_index
          << " doesn't support subgroup extended types, skipping." << std::endl;
      device_index++;
      continue;
    }

    // Other physical features we can error on lazily if they are used in
    // kernels which have the SPIR-V capabilities we don't support. So still
    // create a device but record the relevant features.
    uint16_t backend_features{};
    if (features_12.shaderFloat16) {
      backend_features |= vk_device_features::shaderFloat16;
    }
    if (features_12.storagePushConstant8) {
      backend_features |= vk_device_features::storagePushConstant8;
    }
    if (features_12.shaderInt8) {
      backend_features |= vk_device_features::shaderInt8;
    }
    if (features_12.uniformAndStorageBuffer8BitAccess) {
      backend_features |= vk_device_features::uniformAndStorageBuffer8BitAccess;
    }

    vk::PhysicalDeviceFeatures const &features =
        supported_features.get<vk::PhysicalDeviceFeatures2>().features;
    if (features.shaderInt16) {
      backend_features |= vk_device_features::shaderInt16;
    }
    if (features.shaderInt64) {
      backend_features |= vk_device_features::shaderInt64;
    }
    if (features.shaderFloat64) {
      backend_features |= vk_device_features::shaderFloat64;
    }

    auto const &features_11 =
        supported_features.get<vk::PhysicalDeviceVulkan11Features>();
    if (features_11.variablePointers) {
      backend_features |= vk_device_features::variablePointers;
    }

    if (features_11.variablePointersStorageBuffer) {
      backend_features |= vk_device_features::variablePointersStorageBuffer;
    }

    if (features_11.storagePushConstant16) {
      backend_features |= vk_device_features::storagePushConstant16;
    }

    if (features_11.uniformAndStorageBuffer16BitAccess) {
      backend_features |=
          vk_device_features::uniformAndStorageBuffer16BitAccess;
    }

    auto device_name = phys_dev.getProperties().deviceName;
    if (device_matches(visibility_mask, backend_id::vk, device_index,
                       device_index, 0, device_name, {})) {
      _devices.emplace_back(phys_dev, static_cast<int>(_devices.size()),
                            backend_features);
    }
    device_index++;
  }
  for (auto &dev : _devices) {
    dev.init();
  }
}

vk_hardware_manager::~vk_hardware_manager() {
  // We need to free the objects created from the vkInstance
  // before destroying the vkInstance RAII member
  _devices.clear();
}

std::size_t vk_hardware_manager::get_num_devices() const {
  return _devices.size();
}

hardware_context *vk_hardware_manager::get_device(std::size_t index) {
  return &(_devices[index]);
}

device_id vk_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{backend_descriptor{_hw_platform, api_platform::vk},
                   static_cast<int>(index)};
}

std::size_t vk_hardware_manager::get_num_platforms() const { return 1; }

} // namespace rt
} // namespace hipsycl
