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

#ifndef HIPSYCL_SSCP_AMDGPU_BUILTINS_HPP
#define HIPSYCL_SSCP_AMDGPU_BUILTINS_HPP

#include "builtin_config.hpp"
#include "atomic.hpp"

#include <hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp>
extern "C" int __acpp_amdgpu_dpp_unsupported_on_rdna_or_non_amd();

namespace adaptivecpp::amdgpu {

extern "C" int __acpp_sscp_custom_intrinsic__llvm_amdgcn_update_dpp_i32(
    int old_val, int src, int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl);

template<int dpp_ctrl, int row_mask, int bank_mask, bool bound_ctrl>
inline int update_dpp(int value) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd &&
    jit::reflect<jit::reflection_query::target_arch>() < 0x1000,
    [&]() {
      return __acpp_sscp_custom_intrinsic__llvm_amdgcn_update_dpp_i32(
          0, value, dpp_ctrl, row_mask, bank_mask, bound_ctrl);
    },
    [&]() {
      return __acpp_amdgpu_dpp_unsupported_on_rdna_or_non_amd();
    }
  );
}

HIPSYCL_SSCP_BUILTIN int readfirstlane(int value);

HIPSYCL_SSCP_BUILTIN float fract(float value);

HIPSYCL_SSCP_BUILTIN_ATTRIBUTES float unsafe_atomic_fetch_add(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, float *ptr, float x);

HIPSYCL_SSCP_BUILTIN_ATTRIBUTES double unsafe_atomic_fetch_add(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, double *ptr, double x);

} // namespace adaptivecpp::amdgpu

#endif // HIPSYCL_SSCP_AMDGPU_BUILTINS_HPP
