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
#include "hipSYCL/runtime/kernel_configuration.hpp"

namespace hipsycl {
namespace rt {

namespace {
class string_build_config_mapper {
public:
  string_build_config_mapper() {
    _options =  {
      {"known-group-size-x", kernel_build_option::known_group_size_x},
      {"known-group-size-y", kernel_build_option::known_group_size_y},
      {"known-group-size-z", kernel_build_option::known_group_size_z},
      {"known-local-mem-size", kernel_build_option::known_local_mem_size},
      {"ptx-version", kernel_build_option::ptx_version},
      {"ptx-target-device", kernel_build_option::ptx_target_device},
      {"amdgpu-target-device", kernel_build_option::amdgpu_target_device},
      {"rocm-device-libs-path", kernel_build_option::amdgpu_rocm_device_libs_path},
      {"rocm-path", kernel_build_option::amdgpu_rocm_path},
      {"spirv-dynamic-local-mem-allocation-size", kernel_build_option::spirv_dynamic_local_mem_allocation_size}
    };

    _flags = {
      {"global-sizes-fit-in-int", kernel_build_flag::global_sizes_fit_in_int},
      {"fast-math", kernel_build_flag::fast_math},
      {"ptx-ftz", kernel_build_flag::ptx_ftz},
      {"ptx-approx-div", kernel_build_flag::ptx_approx_div},
      {"ptx-approx-sqrt", kernel_build_flag::ptx_approx_sqrt},
      {"spirv-enable-intel-llvm-spirv-options", kernel_build_flag::spirv_enable_intel_llvm_spirv_options}
    };

    for(const auto& elem : _options) {
      _inverse_options[elem.second] = elem.first;
    }

    for(const auto& elem : _flags) {
      _inverse_flags[elem.second] = elem.first;
    }
  }

  const auto&
  string_to_build_option_map() const{
    return _options;
  }

  const auto&
  string_to_build_flag_map() const {
    return _flags;
  }

  const auto&
  build_option_to_string_map() const {
    return _inverse_options;
  }

  const auto&
  build_flag_to_string_map() const {
    return _inverse_flags;
  }

  static string_build_config_mapper& get() {
    static string_build_config_mapper mapper;
    return mapper;
  }
private:

  std::unordered_map<std::string, kernel_build_option> _options;
  std::unordered_map<std::string, kernel_build_flag> _flags;
  std::unordered_map<kernel_build_option, std::string> _inverse_options;
  std::unordered_map<kernel_build_flag, std::string> _inverse_flags;
};

}



std::string to_string(kernel_build_flag f) {
  const auto& map = string_build_config_mapper::get().build_flag_to_string_map();
  auto it = map.find(f);
  if(it == map.end())
    return {};
  return it->second;
}

std::string to_string(kernel_build_option o) {
  const auto& map = string_build_config_mapper::get().build_option_to_string_map();
  auto it = map.find(o);
  if(it == map.end())
    return {};
  return it->second;
}

std::optional<kernel_build_option>
to_build_option(const std::string& s) {
  const auto& map = string_build_config_mapper::get().string_to_build_option_map();
  auto it = map.find(s);
  if(it == map.end())
    return {};
  return it->second;
}

std::optional<kernel_build_flag>
to_build_flag(const std::string& s) {
  const auto& map = string_build_config_mapper::get().string_to_build_flag_map();
  auto it = map.find(s);
  if(it == map.end())
    return {};
  return it->second;
}


}
}

