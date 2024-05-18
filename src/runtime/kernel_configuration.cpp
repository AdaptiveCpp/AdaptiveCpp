/*
 * This file is part of hipSYCL, a SYCL implementation based on OMP/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
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

