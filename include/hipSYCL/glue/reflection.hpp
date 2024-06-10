/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_REFLECTION_HPP
#define HIPSYCL_REFLECTION_HPP

#include <mutex>
#include <unordered_map>

template <class StructT>
void __acpp_introspect_flattened_struct(void *s, int **num_flattened_members,
                                        int **member_offsets,
                                        int **member_sizes, int **member_kinds);

namespace hipsycl::glue::reflection {

enum class type_kind : int { other = 0, pointer = 1, integer_type = 2, float_type = 3 };

class introspect_flattened_struct {
public:
  template<class StructT>
  introspect_flattened_struct(const StructT& s) 
  : _num_members{nullptr} {
    // Compiler needs this copy to figure out
    // the struct type in the presence of opaque pointers.
    // Currently it checks if the first operand of the call
    // to the builtin comes from an alloca instruction.
    StructT s_copy = s;
    __acpp_introspect_flattened_struct<StructT>(&s_copy, &_num_members,
                                                   &_member_offsets, &_member_sizes,
                                                   reinterpret_cast<int **>(&_member_kinds));
  }

  int get_num_members() const {
    if (!_num_members)
      return 0;
    return *_num_members;
  }

  int get_member_offset(int idx) const {
    return _member_offsets[idx];
  }

  int get_member_size(int idx) const {
    return _member_sizes[idx];
  }

  type_kind get_member_kind(int idx) const {
    return _member_kinds[idx];
  }
private:
  int* _num_members;
  int* _member_offsets;
  int* _member_sizes;
  type_kind* _member_kinds;
};

namespace detail {

class symbol_information {
public:
  static symbol_information& get() {
    static symbol_information si;
    return si;
  }

  void register_function_symbol(const void* address, const char* name) {
    std::lock_guard<std::mutex> lock{_mutex};
    _symbol_names[address] = name;
  }

  const char* resolve_symbol_name(const void* address) const {
    std::lock_guard<std::mutex> lock{_mutex};
    auto it = _symbol_names.find(address);
    if(it == _symbol_names.end())
      return nullptr;
    return it->second;
  }
private:
  symbol_information() = default;
  std::unordered_map<const void*, const char*> _symbol_names;
  mutable std::mutex _mutex;
};

}

template<class Ret, typename... Args>
const char* resolve_function_name(Ret (*func)(Args...)) {
  return detail::symbol_information::get().resolve_symbol_name((void*)func);
}


}


// Will be defined by compiler
extern "C" void __acpp_reflection_init_registered_function_pointers();
extern "C" void __acpp_function_annotation_needs_function_ptr_argument_reflection();

// Compiler will generate calls to this function to register functions.
__attribute__((used))
inline void __acpp_reflection_associate_function_pointer(const void *func_ptr,
                                             const char *func_name) {
  hipsycl::glue::reflection::detail::symbol_information::get()
      .register_function_symbol(func_ptr, func_name);
}

struct __acpp_reflection_tu_init_trigger {
  __acpp_reflection_tu_init_trigger() {
    __acpp_reflection_init_registered_function_pointers();
  }
};

static __acpp_reflection_tu_init_trigger __reflection_init;


namespace hipsycl::glue::reflection {

template<class Ret, typename... Args>
__attribute__((noinline))
void enable_function_symbol_reflection(Ret (*func)(Args...)) {
  __acpp_function_annotation_needs_function_ptr_argument_reflection();
}

}

#endif
