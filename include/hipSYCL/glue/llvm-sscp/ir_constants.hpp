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
#ifndef HIPSYCL_IR_CONSTANTS_HPP
#define HIPSYCL_IR_CONSTANTS_HPP

#include <type_traits>

#include "s1_ir_constants.hpp"
#include "s2_ir_constants.hpp"

template <auto &ConstantName, class ValueT>
ValueT __acpp_sscp_s2_ir_constant<ConstantName, ValueT>::get(
    ValueT default_value) noexcept {
  // The static variable will cause clang to emit a global variable in LLVM IR,
  // that we will turn into a constant during S2 compilation.
  //
  // TODO We may have to suppress compiler warnings about uninitialized data
  // here
  //
  // S2 Compiler will look for special identifier __acpp_ir_constant_v to
  // distinguish the actual IR constant from other global variables related to
  // this class (e.g. type information).
  static ValueT __acpp_ir_constant_v;
  if (__acpp_sscp_is_device) {
    return __acpp_ir_constant_v;
  } else {
    return default_value;
  }
}

namespace hipsycl::sycl::jit {

template <auto &ConstantName, class ValueT>
auto introspect(ValueT default_value = {}) noexcept {
  return __acpp_sscp_s2_ir_constant<ConstantName, ValueT>::get(default_value);
}
}

#endif
