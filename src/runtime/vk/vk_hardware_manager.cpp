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
#include <algorithm>
#include <limits>
#include <sstream>

namespace hipsycl {
namespace rt {

vk_hardware_context::vk_hardware_context(
    const vk::raii::PhysicalDevice &phys_device, int dev_id, uint16_t features,
    bool portability_subset)
    : _physical_device(phys_device), _dev_id(dev_id),
      _physical_dev_features(features) {
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
  // Outside if statement scope so lifetime is valid for raii::Device
  // initialization
  const char *portability_ext_name = VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME;
  if (portability_subset) {
    dev_create_info.setEnabledExtensionCount(1);
    dev_create_info.setPpEnabledExtensionNames(&portability_ext_name);
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

  {
    std::stringstream ss;
    ss << "vk_hardware_context: logical device constructed "
       << "for device id " << dev_id << " with queue family "
       << "index " << _queue_index << std::endl;
    HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
  }
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
  case device_support_aspect::execution_timestamps:
    return false;
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
  }
  assert(false && "Unknown device aspect");
  std::terminate();
}

std::size_t vk_hardware_context::get_property(device_uint_property prop) const {
  switch (prop) {
  case device_uint_property::max_compute_units:
    return 1; // No matching VK query
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
  case device_uint_property::preferred_vector_width_char:
  case device_uint_property::preferred_vector_width_double:
  case device_uint_property::preferred_vector_width_float:
  case device_uint_property::preferred_vector_width_half:
  case device_uint_property::preferred_vector_width_int:
  case device_uint_property::preferred_vector_width_long:
  case device_uint_property::preferred_vector_width_short:
  case device_uint_property::native_vector_width_char:
  case device_uint_property::native_vector_width_double:
  case device_uint_property::native_vector_width_float:
  case device_uint_property::native_vector_width_half:
  case device_uint_property::native_vector_width_int:
  case device_uint_property::native_vector_width_long:
  case device_uint_property::native_vector_width_short:
    return 1; // TODO - figure out mapping
  case device_uint_property::max_clock_speed:
    return 0; // TODO - figure out mapping
  case device_uint_property::max_malloc_size:
    return std::min(_allocator->get_global_mem_size(), _max_alloc_size);
  case device_uint_property::address_bits:
    return 64; // spirv64 only for now
  // TODO - Don't support images for now
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
    return 0;
  case device_uint_property::max_parameter_size:
    // TODO - figure out mapping
    return std::numeric_limits<std::size_t>::max();
  case device_uint_property::mem_base_addr_align:
    return 0; // TODO - figure out mapping
  case device_uint_property::global_mem_cache_line_size:
    return _limits.nonCoherentAtomSize;
  case device_uint_property::global_mem_cache_size:
    return 0; // TODO - figure out mapping
  case device_uint_property::global_mem_size:
    return _allocator->get_global_mem_size();
  case device_uint_property::max_constant_buffer_size:
    return 0; // TODO - figure out mapping
  case device_uint_property::max_constant_args:
    // TODO - figure out mapping
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
    return 0; // TODO - figure out mapping
  case device_uint_property::backend_id:
    return static_cast<int>(backend_id::vk);
  case device_uint_property::queue_priority_range_low:
  case device_uint_property::queue_priority_range_high:
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
  // TODO - Not all vendors will use driver versions in this form
  uint32_t version = _properties.driverVersion;
  uint32_t major = VK_API_VERSION_MAJOR(version);
  uint32_t minor = VK_API_VERSION_MINOR(version);
  uint32_t patch = VK_API_VERSION_PATCH(version);
  return std::to_string(major) + "." + std::to_string(minor) + "." +
         std::to_string(patch);
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

  auto optional_ext = VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME;
  bool optional_ext_supported = std::any_of(
      std::begin(ext_properties), std::end(ext_properties),
      [optional_ext](auto const &ext_property) {
        return strcmp(ext_property.extensionName, optional_ext) == 0;
      });
  if (optional_ext_supported) {
    {
      std::stringstream ss;
      ss << "vk_hardware_manager: enabling extension " << optional_ext
         << std::endl;
      HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
    }
    enabled_extensions.push_back(optional_ext);
  }

  // To layer ontop of moltenVK we need to opt-in to the Vulkan loader showing
  // non-conformant Vulkan implementations
  const vk::InstanceCreateFlags flags =
      vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
  vk::InstanceCreateInfo create_info{
      flags,
      &app_info,                                        // pApplicationInfo
      static_cast<uint32_t>(required_layers.size()),    // enabledLayerCount
      required_layers.data(),                           // ppEnabledLayerNames
      static_cast<uint32_t>(enabled_extensions.size()), // enabledExtensionCount
      enabled_extensions.data() // ppEnabledExtensionNames
  };
  _instance = vk::raii::Instance(_context, create_info);

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

  _debug_messenger =
      _instance.createDebugUtilsMessengerEXT(debug_utils_messenger_create_info);
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
      {
        std::stringstream ss;
        ss << "vk_hardware_manager: physical device " << device_index
           << "doesn't support Vulkan 1.2, skipping." << std::endl;
        HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
      }
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
      {
        std::stringstream ss;
        ss << "vk_hardware_manager: physical device " << device_index
           << "doesn't support compute queue, skipping." << std::endl;
        HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
      }
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
      {
        std::stringstream ss;
        ss << "vk_hardware_manager: physical device " << device_index
           << "doesn't support bufferDeviceAddress, skipping." << std::endl;
        HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
      }
      device_index++;
      continue;
    }
    // Essential for synchronization, don't create a backend device without it.
    if (!features_12.timelineSemaphore) {
      {
        std::stringstream ss;
        ss << "vk_hardware_manager: physical device " << device_index
           << "doesn't support timeline semaphore, skipping." << std::endl;
        HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
      }
      device_index++;
      continue;
    }
    if (!features_12.shaderSubgroupExtendedTypes) {
      {
        std::stringstream ss;
        ss << "vk_hardware_manager: physical device " << device_index
           << "doesn't support subgroup extended types, skipping." << std::endl;
        HIPSYCL_DEBUG_INFO_ATOMIC(ss.rdbuf());
      }
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

    auto phys_dev_extensions = phys_dev.enumerateDeviceExtensionProperties();
    bool portability_ext_supported = std::any_of(
        std::begin(phys_dev_extensions), std::end(phys_dev_extensions),
        [](auto const &ext_property) {
          return strcmp(ext_property.extensionName,
                        VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME) == 0;
        });

    auto device_name = phys_dev.getProperties().deviceName;
    if (device_matches(visibility_mask, backend_id::vk, device_index,
                       device_index, 0, device_name, {})) {
      _devices.emplace_back(phys_dev, device_index, backend_features,
                            portability_ext_supported);
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
