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
// __hipsycl_sscp_s2_ir_constant
template<auto& ConstantName, class ValueT>
struct __hipsycl_sscp_s2_ir_constant {
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
