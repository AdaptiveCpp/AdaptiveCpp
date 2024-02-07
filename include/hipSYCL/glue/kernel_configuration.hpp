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
#include "hipSYCL/common/stable_running_hash.hpp"

namespace hipsycl {
namespace glue {

enum class kernel_base_config_parameter : int {
  backend_id = 0,
  compilation_flow = 1,
  hcf_object_id = 2,
  target_arch = 3,
  runtime_device = 4,
  runtime_context = 5
};

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

  void set_build_option(const std::string& option, const std::string& value) {
    _build_options.push_back(std::make_pair(option, value));
  }

  template<class T>
  void set_build_option(const std::string& option, const T& value) {
    set_build_option(option, std::to_string(value));
  }

  void set_build_flag(const std::string& flag) {
    _build_flags.push_back(flag);
  }

  template<class KeyT, class ValueT>
  void append_base_configuration(const KeyT key, const ValueT& value) {
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
      add_entry_to_hash(result, entry.first.data(), entry.first.size(),
                        entry.second.data(), entry.second.size());
    }

    for(const auto& entry : _build_flags) {
      add_entry_to_hash(result, entry.data(), entry.size(),
                        "", 0);
    }

    return result;
  }

  const std::vector<s2_ir_configuration_entry>& s2_ir_entries() const {
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
  std::vector<std::string> _build_flags;
  std::vector<std::pair<std::string, std::string>> _build_options;

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
