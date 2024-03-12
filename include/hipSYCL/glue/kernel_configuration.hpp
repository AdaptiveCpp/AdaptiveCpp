/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#ifndef HIPSYCL_KERNEL_CONFIGURATION_HPP
#define HIPSYCL_KERNEL_CONFIGURATION_HPP

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <array>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>
#include <functional>
#include <cassert>
#include <optional>
#include <unordered_map>

#include "hipSYCL/common/stable_running_hash.hpp"


namespace hipsycl {
namespace glue {

enum class kernel_base_config_parameter : int {
  backend_id = 0,
  compilation_flow = 1,
  hcf_object_id = 2,
  target_arch = 3,
  runtime_device = 4,
  runtime_context = 5,
  single_kernel = 6
};

enum class kernel_build_option : int {
  known_group_size_x,
  known_group_size_y,
  known_group_size_z,
  known_local_mem_size,

  ptx_version,
  ptx_target_device,

  amdgpu_target_device,
  amdgpu_rocm_device_libs_path,
  amdgpu_rocm_path,

  spirv_dynamic_local_mem_allocation_size
};

enum class kernel_build_flag : int {
  global_sizes_fit_in_int,
  fast_math,

  ptx_ftz,
  ptx_approx_div,
  ptx_approx_sqrt,

  spirv_enable_intel_llvm_spirv_options
};

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

  static const auto& string_to_build_option_map() {
    return get()._options;
  }

  static const auto& string_to_build_flag_map() {
    return get()._flags;
  }

  static const auto& build_option_to_string_map() {
    return get()._inverse_options;
  }

  static const auto& build_flag_to_string_map() {
    return get()._inverse_flags;
  }
private:
  static string_build_config_mapper& get() {
    static string_build_config_mapper mapper;
    return mapper;
  }

  std::unordered_map<std::string, kernel_build_option> _options;
  std::unordered_map<std::string, kernel_build_flag> _flags;
  std::unordered_map<kernel_build_option, std::string> _inverse_options;
  std::unordered_map<kernel_build_flag, std::string> _inverse_flags;
};

inline std::string to_string(kernel_build_flag f) {
  const auto& map = string_build_config_mapper::build_flag_to_string_map();
  auto it = map.find(f);
  if(it == map.end())
    return {};
  return it->second;
}

inline std::string to_string(kernel_build_option o) {
  const auto& map = string_build_config_mapper::build_option_to_string_map();
  auto it = map.find(o);
  if(it == map.end())
    return {};
  return it->second;
}

inline std::optional<kernel_build_option>
to_build_option(const std::string& s) {
  const auto& map = string_build_config_mapper::string_to_build_option_map();
  auto it = map.find(s);
  if(it == map.end())
    return {};
  return it->second;
}

inline std::optional<kernel_build_flag>
to_build_flag(const std::string& s) {
  const auto& map = string_build_config_mapper::string_to_build_flag_map();
  auto it = map.find(s);
  if(it == map.end())
    return {};
  return it->second;
}

class kernel_configuration {

  class s2_ir_configuration_entry {
    static constexpr std::size_t buffer_size = 8;

    std::string _name;
    std::type_index _type;
    std::array<int8_t, buffer_size> _value;
    std::size_t _data_size;
    

    template<class T>
    void store(const T& val) {
      static_assert(sizeof(T) <= buffer_size,
                    "Unsupported kernel configuration value type");
      for(int i = 0; i < _value.size(); ++i)
        _value[i] = 0;
      
      memcpy(_value.data(), &val, sizeof(val));
    }

  public:
    template<class T>
    s2_ir_configuration_entry(const std::string& name, const T& val)
    : _name{name}, _type{typeid(T)}, _data_size{sizeof(T)} {
      store<T>(val);
    }

    template<class T>
    T get_value() const {
      static_assert(sizeof(T) <= buffer_size,
                    "Unsupported kernel configuration value type");
      T v;
      memcpy(&v, _value.data(), sizeof(T));
      return v;
    }

    template<class T>
    bool is_type() const {
      return _type == typeid(T);
    }

    const void* get_data_buffer() const {
      return _value.data();
    }

    std::size_t get_data_size() const {
      return _data_size;
    }

    const std::string& get_name() const {
      return _name;
    }
  };



public:
  struct int_or_string{
    std::optional<uint64_t> int_value;
    std::optional<std::string> string_value;
  };

  using id_type = std::array<uint64_t, 2>;

  template<class T>
  void set_s2_ir_constant(const std::string& config_parameter_name, const T& value) {
    s2_ir_configuration_entry entry{config_parameter_name, value};
    for(int i = 0; i < _s2_ir_configurations.size(); ++i) {
      if(_s2_ir_configurations[i].get_name() == config_parameter_name) {
        _s2_ir_configurations[i] = entry;
        return;
      }
    }
    _s2_ir_configurations.push_back(entry);
  }

  void set_build_option(kernel_build_option option, const std::string& value) {
    int_or_string ios;
    ios.string_value = value;
    _build_options.push_back(std::make_pair(option, ios));
  }

  template<class T, std::enable_if_t<std::is_unsigned_v<T>, int> = 0>
  void set_build_option(kernel_build_option option, T int_value) {
    int_or_string ios;
    ios.int_value = static_cast<uint64_t>(int_value);
    _build_options.push_back(std::make_pair(option, ios));
  }

  template<class T, std::enable_if_t<!std::is_unsigned_v<T>, int> = 0>
  void set_build_option(kernel_build_option option, const T& value) {
    set_build_option(option, std::to_string(value));
  }

  void set_build_flag(kernel_build_flag flag) {
    _build_flags.push_back(flag);
  }

  template <class ValueT>
  void append_base_configuration(kernel_base_config_parameter key,
                                 const ValueT &value) {
    add_entry_to_hash(_base_configuration_result, data_ptr(key), data_size(key),
                      data_ptr(value), data_size(value));
  }

  template<class KeyT, class ValueT>
  static void extend_hash(id_type& hash, const KeyT& key, const ValueT& value) {
    add_entry_to_hash(hash, data_ptr(key), data_size(key),
                      data_ptr(value), data_size(value));
  }

  static std::string to_string(const id_type& id) {
    return std::to_string(id[0])+"."+std::to_string(id[1]);
  }

  id_type generate_id() const {
    id_type result = _base_configuration_result;

    for(const auto& entry : _s2_ir_configurations) {
      add_entry_to_hash(result, entry.get_name().data(),
                        entry.get_name().size(), entry.get_data_buffer(),
                        entry.get_data_size());
    }

    for(const auto& entry : _build_options) {
      uint64_t numeric_option_id = static_cast<uint64_t>(entry.first) | (1ull << 32);
      if(entry.second.int_value.has_value()) {
        auto numeric_value = entry.second.int_value.value();
        add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        &numeric_value, sizeof(numeric_value));
      } else {
        const std::string& string_value = entry.second.string_value.value();
        add_entry_to_hash(result, &numeric_option_id, sizeof(numeric_option_id),
                        string_value.data(), string_value.size());
      }
    }

    for(const auto& entry : _build_flags) {
      uint64_t numeric_flag_id = static_cast<uint64_t>(entry) | (1ull << 33);
      add_entry_to_hash(result, &numeric_flag_id, sizeof(numeric_flag_id),
                        "", 0);
    }

    return result;
  }

  const auto& s2_ir_entries() const {
    return _s2_ir_configurations;
  }

  const auto& build_options() const {
    return _build_options;
  }

  const auto& build_flags() const {
    return _build_flags;
  }

private:
  static const void* data_ptr(const char* data) {
    return data_ptr(std::string{data});
  }

  static const void* data_ptr(const std::string& data) {
    return data.data();
  }

  template<class T>
  static const void* data_ptr(const std::vector<T>& data) {
    return data.data();
  }

  template<class T>
  static const void* data_ptr(const T& data) {
    return &data;
  }

  static std::size_t data_size(const char* data) {
    return data_size(std::string{data});
  }

  static std::size_t data_size(const std::string& data) {
    return data.size();
  }

  template<class T>
  static std::size_t data_size(const std::vector<T>& data) {
    return data.size();
  }

  template<class T>
  static std::size_t data_size(const T& data) {
    return sizeof(T);
  }

  static void add_entry_to_hash(id_type &hash, const void *key_data,
                         std::size_t key_size, const void *data,
                         std::size_t data_size) {
    common::stable_running_hash h;
    h(key_data, key_size);
    h(data, data_size);
    auto entry_hash = h.get_current_hash();
    hash[entry_hash % hash.size()] ^= entry_hash;
  }


  std::vector<s2_ir_configuration_entry> _s2_ir_configurations;
  std::vector<kernel_build_flag> _build_flags;
  std::vector<std::pair<kernel_build_option, int_or_string>> _build_options;

  id_type _base_configuration_result = {};
};

struct kernel_id_hash{
  std::size_t operator() (const kernel_configuration::id_type &id) const {
    std::size_t hash = 0;
    for(std::size_t i = 0; i < id.size(); ++i)
      hash ^= std::hash<kernel_configuration::id_type::value_type>{}(id[i]);
    return hash;
  }
};

}
}

#endif
