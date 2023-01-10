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
#include <typeindex>
#include <vector>
#include <functional>
#include <cassert>
#include "hipSYCL/common/stable_running_hash.hpp"

namespace hipsycl {
namespace glue {

class kernel_configuration {

  class configuration_entry {
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
    configuration_entry(const std::string& name, const T& val)
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
  void set(const std::string& config_parameter_name, const T& value) {
    configuration_entry entry{config_parameter_name, value};
    for(int i = 0; i < _configurations.size(); ++i) {
      if(_configurations[i].get_name() == config_parameter_name) {
        _configurations[i] = entry;
        return;
      }
    }
    _configurations.push_back(entry);
  }

  id_type generate_id() const {
    id_type result {};

    for(const auto& entry : _configurations) {
      common::stable_running_hash h;
      h(entry.get_name().data(), entry.get_name().size());
      h(entry.get_data_buffer(), entry.get_data_size());

      auto entry_hash = h.get_current_hash();

      result[entry_hash % result.size()] ^= entry_hash;
    }

    return result;
  }

  const std::vector<configuration_entry>& entries() const {
    return _configurations;
  }

  std::vector<configuration_entry> _configurations;
};

}
}

#endif
