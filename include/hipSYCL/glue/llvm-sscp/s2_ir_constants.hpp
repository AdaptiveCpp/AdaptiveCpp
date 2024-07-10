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
#ifndef HIPSYCL_S2_IR_CONSTANTS_HPP
#define HIPSYCL_S2_IR_CONSTANTS_HPP

/// \brief This file contains S2 IR constant definitions that may
/// be shared across the hipSYCL compiler code. 
///
/// As such, no undefined globals should be pulled into this file.
///
/// Unlike Stage 1 IR constants, Stage 2 IR constants can be constructed
/// programmatically by the user.

// S2 IR constants can be identified from their usage of
// __acpp_sscp_s2_ir_constant
template<auto& ConstantName, class ValueT>
struct __acpp_sscp_s2_ir_constant {
  static ValueT get(ValueT default_value) noexcept;

  using value_type = ValueT;
};


namespace hipsycl::glue::sscp {
  struct ir_constant_name {};
}

namespace hipsycl::sycl::jit {

namespace backend {

inline constexpr int spirv = 0;
inline constexpr int ptx = 1;
inline constexpr int amdgpu = 2;
inline constexpr int host = 3;

}

constexpr glue::sscp::ir_constant_name current_backend;

}

#endif
