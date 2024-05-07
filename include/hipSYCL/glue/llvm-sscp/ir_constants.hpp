/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_IR_CONSTANTS_HPP
#define HIPSYCL_IR_CONSTANTS_HPP

#include <type_traits>

#include "s1_ir_constants.hpp"
#include "s2_ir_constants.hpp"

template <auto &ConstantName, class ValueT>
ValueT __hipsycl_sscp_s2_ir_constant<ConstantName, ValueT>::get(
    ValueT default_value) noexcept {
  // The static variable will cause clang to emit a global variable in LLVM IR,
  // that we will turn into a constant during S2 compilation.
  //
  // TODO We may have to suppress compiler warnings about uninitialized data
  // here
  //
  // S2 Compiler will look for special identifier __hipsycl_ir_constant_v to
  // distinguish the actual IR constant from other global variables related to
  // this class (e.g. type information).
  static ValueT __hipsycl_ir_constant_v;
  if (__hipsycl_sscp_is_device) {
    return __hipsycl_ir_constant_v;
  } else {
    return default_value;
  }
}

namespace hipsycl::sycl::jit {

template <auto &ConstantName, class ValueT>
auto introspect(ValueT default_value = {}) noexcept {
  return __hipsycl_sscp_s2_ir_constant<ConstantName, ValueT>::get(default_value);
}
}

#endif
