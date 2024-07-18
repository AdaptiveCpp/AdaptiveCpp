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
#include <unordered_map>

#include "backend.hpp"

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/glue/reflection.hpp"
#include "hipSYCL/glue/llvm-sscp/fcall_specialization.hpp"
#include "hipSYCL/common/stable_running_hash.hpp"
#include "hipSYCL/common/unordered_dense.hpp"
#include "exception.hpp"

// Meaning: The calling function has a dynamic function as argument 0 (excluding this pointer)
extern "C" void __acpp_function_annotation_dynamic_function();

// Meaning: The calling function has a dynamic function as argument 0 (excluding this pointer)
extern "C" void __acpp_function_annotation_dynamic_function_def_arg0();
// Meaning: The calling function has a dynamic function as argument 1 (excluding this pointer)
extern "C" void __acpp_function_annotation_dynamic_function_def_arg1();

template<class T>
void __acpp_function_annotation_argument_used(T&& x);

namespace hipsycl::sycl::jit {

template<class T>
void arguments_are_used(T&& x) {
  __acpp_function_annotation_argument_used(std::forward<T>(x));
}

template<class T, typename... Args>
void arguments_are_used(T&& x, Args&&... other_args) {
  __acpp_function_annotation_argument_used(std::forward<T>(x));
  arguments_are_used(std::forward<Args>(other_args)...);
}

class dynamic_function_id {
public:
  struct __handle {};

  dynamic_function_id() = default;
  explicit dynamic_function_id(const __handle* handle)
  : _id{handle} {}

  const __handle* get_handle() const {
    return _id;
  }

private:
  const __handle* _id;
};


template<class Ret, typename... Args>
class dynamic_function {
public:
  [[clang::noinline]]
  dynamic_function(Ret (*func)(Args...)) {
    // clang::annotate attributes are not reliably emitted
    // to IR for class member functions, so we use these
    // annotation functions to convey this information instead.
    __acpp_function_annotation_dynamic_function();
    __acpp_function_annotation_needs_function_ptr_argument_reflection();

    _function_name = glue::reflection::resolve_function_name(func);
    if(!_function_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function: Could not resolve "
                            "function symbol"};
  }

  dynamic_function_id id() const {
    return dynamic_function_id{
        reinterpret_cast<const dynamic_function_id::__handle *>(
            _function_name)};
  }

  const char* function_name() const {
    return _function_name;
  }
private:
  const char* _function_name;
};


template<class Ret, typename... Args>
class dynamic_function_definition {
public:

  [[clang::noinline]]
  dynamic_function_definition(Ret (*func)(Args...)) {
    __acpp_function_annotation_dynamic_function_def_arg0();
    __acpp_function_annotation_needs_function_ptr_argument_reflection();

    _function_name = glue::reflection::resolve_function_name(func);
    if(!_function_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_definition: Could not resolve "
                            "function symbol"};
  }

  dynamic_function_id id() const {
    return dynamic_function_id{
        reinterpret_cast<const dynamic_function_id::__handle *>(
            _function_name)};
  }

  const char* function_name() const {
    return _function_name;
  }
private:
  const char* _function_name;
};



class dynamic_function_config {
public:

  void define(dynamic_function_id function, dynamic_function_id definition) {
    _is_ready = false;

    const char* function_name = reinterpret_cast<const char*>(function.get_handle());
    const char* definition_name = reinterpret_cast<const char*>(definition.get_handle());

    if(!function_name || !definition_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_config: Invalid dynamic_function_id"};

    set_entry(function_name, std::vector<std::string>{definition_name});
  }

  template <class Ret, typename... Args>
  void define(dynamic_function<Ret, Args...> df,
              dynamic_function_definition<Ret, Args...> definition) {
    define(df.id(), definition.id());
  }

  template <class Ret, typename... Args>
  [[clang::noinline]]
  void define(Ret (*df)(Args...),
              Ret (*definition)(Args...)) {
    __acpp_function_annotation_dynamic_function();
    __acpp_function_annotation_dynamic_function_def_arg1();
    __acpp_function_annotation_needs_function_ptr_argument_reflection();

    define(pointer_to_dynamic_function_id(df), pointer_to_dynamic_function_id(definition));
  }

  void
  define_as_call_sequence(dynamic_function_id call,
                          const std::vector<dynamic_function_id> &definitions) {
    _is_ready = false;

    const char* function_name = reinterpret_cast<const char*>(call.get_handle());
    if(!function_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_config: Invalid dynamic_function_id"};
    
    std::vector<std::string> definition_names;
    for(auto r : definitions) {
      const char* definition_name = reinterpret_cast<const char*>(r.get_handle());

      if(!definition_name) {
        throw sycl::exception{make_error_code(errc::invalid),
                              "dynamic_function_config: Invalid "
                              "dynamic_function_id for definition"};
      }
      definition_names.emplace_back(std::string{definition_name});
    }

    set_entry(function_name, definition_names);
  }

  template <typename... Args>
  void define_as_call_sequence(
      dynamic_function<void, Args...> call,
      const std::vector<dynamic_function_definition<void, Args...>>
          &definitions) {
    
    std::vector<dynamic_function_id> definition_ids;
    definition_ids.reserve(definitions.size());
    for(const auto& d : definitions)
      definition_ids.push_back(d.id());
    
    define_as_call_sequence(call.id(), definition_ids);
  }


  template <typename... Args>
  [[clang::noinline]]
  void define_as_call_sequence(
      void (*call)(Args...),
      const std::vector<dynamic_function_definition<void, Args...>>
          &definitions) {
    __acpp_function_annotation_dynamic_function();
    __acpp_function_annotation_needs_function_ptr_argument_reflection();
    
    std::vector<dynamic_function_id> definition_ids;
    definition_ids.reserve(definitions.size());
    for(const auto& d : definitions)
      definition_ids.push_back(d.id());
    
    define_as_call_sequence(pointer_to_dynamic_function_id(call), definition_ids);
  }

  template<class Kernel>
  auto apply(Kernel k) {
    if(!_is_ready)
      prepare_for_submission();

    glue::sscp::fcall_config_kernel_property_t prop{&_config};
    return [prop, k](auto&&... args){
      k(decltype(args)(args)...);
    };
  }

private:

  void set_entry(const char* function_name, const std::vector<std::string>& data) {
    _entries[function_name] = data;
  }

  template<class Ret, typename... Args>
  dynamic_function_id pointer_to_dynamic_function_id(Ret (*func)(Args...)) const {
    const char* function_name = glue::reflection::resolve_function_name(func);
    if(!function_name)
      throw sycl::exception{make_error_code(errc::invalid),
                            "dynamic_function_config: Could not resolve "
                            "function symbol"};
    return dynamic_function_id{
        reinterpret_cast<const dynamic_function_id::__handle *>(
            function_name)};
  }
  
  void autogenerate_fcall_id() {
    _config.unique_hash = 0;
    for(auto& entry : _config.function_call_map) {
      common::stable_running_hash hash;
      hash(entry.first.data(), entry.first.size());
      for(const auto& s : entry.second)
        hash(entry.second.data(), entry.second.size());
      _config.unique_hash ^= hash.get_current_hash();
    }
  }

  void prepare_for_submission() {
    _config.function_call_map.clear();
    for(const auto& entry : _entries) {
      _config.function_call_map.push_back(entry);
    }
    autogenerate_fcall_id();
    _is_ready = true;
  }

  bool _is_ready = false;
  glue::sscp::fcall_specialized_config _config;
  ankerl::unordered_dense::map<const char*, std::vector<std::string>> _entries;
};

}


#endif // IS_DEVICE_PASS_SSCP

#endif
