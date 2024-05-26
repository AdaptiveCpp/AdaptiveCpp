/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
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

#ifndef HIPSYCL_SYCL_JIT_HPP
#define HIPSYCL_SYCL_JIT_HPP

#include <vector>

#include "backend.hpp"

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/glue/reflection.hpp"
#include "hipSYCL/glue/llvm-sscp/fcall_specialization.hpp"
#include "hipSYCL/common/stable_running_hash.hpp"
#include "exception.hpp"

namespace hipsycl::sycl::jit {

class dynamic_function_config {
public:

  template<class Ret, typename... Args>
  void define(Ret (*call)(Args...), Ret(*definition)(Args...)) {
    _has_id = false;

    const char* function_name = glue::reflection::resolve_function_name(call);
    const char* definition_name = glue::reflection::resolve_function_name(definition);

    if(!function_name || !definition_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_definition: Could not resolve "
                            "function symbol"};

    // TODO: What should happen if we already have an entry for that function?

    _config.function_call_map.push_back(std::make_pair(
        std::string(function_name), std::vector<std::string>{definition_name}));
  }

  template <typename... Args>
  void define_as_call_sequence(void (*call)(Args...),
                          const std::vector<void (*)(Args...)> &definitions) {
    _has_id = false;

    const char* function_name = glue::reflection::resolve_function_name(call);
    if(!function_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_definition: Could not resolve "
                            "function symbol"};

    std::vector<std::string> definition_names;
    for(auto r : definitions) {
      const char* definition_name = glue::reflection::resolve_function_name(r);

      if(!definition_name) {
        throw sycl::exception{make_error_code(errc::invalid),
                              "dynamic_function_definition: Could not resolve "
                              "function symbol"};
      }
      definition_names.emplace_back(std::string{definition_name});
    }
    _config.function_call_map.push_back(
        std::make_pair(std::string{function_name}, definition_names));
  }

  template<class Kernel>
  auto apply(Kernel k) {
    if(!_has_id)
      autogenerate_fcall_id();

    glue::sscp::fcall_config_kernel_property_t prop{&_config};
    return [prop, k](auto&&... args){
      k(decltype(args)(args)...);
    };
  }

private:
  
  void autogenerate_fcall_id() {
    for(auto& entry : _config.function_call_map) {
      common::stable_running_hash hash;
      hash(entry.first.data(), entry.first.size());
      for(const auto& s : entry.second)
        hash(entry.second.data(), entry.second.size());
      _config.unique_hash ^= hash.get_current_hash();
    }
    _has_id = true;
  }

  bool _has_id = false;
  glue::sscp::fcall_specialized_config _config;
  
};

}

#endif // IS_DEVICE_PASS_SSCP

#endif
