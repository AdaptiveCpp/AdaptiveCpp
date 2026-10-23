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
#include <hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp>

// Include our auto-generated Clang wrappers which bypass the need for TargetSpecificIRMapper!
// This file is generated at build time by src/compiler/builtin_gen/main.cpp
#include "amdgpu_auto_builtins.hpp"

extern "C" int __acpp_amdgpu_builtin_unsupported_on_non_amd();
extern "C" float __acpp_amdgpu_fract_unsupported_on_non_amd();

namespace adaptivecpp::amdgpu {

static constexpr int kGfx90aArchId = 0x90a;


inline int readfirstlane(int value) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd,
    [&]() {
      return __acpp___builtin_amdgcn_readfirstlane(value);
    },
    [&]() {
      return __acpp_amdgpu_builtin_unsupported_on_non_amd();
    }
  );
}

inline float fract(float value) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd,
    [&]() {
      return __acpp___builtin_amdgcn_fractf(value);
    },
    [&]() {
      return __acpp_amdgpu_fract_unsupported_on_non_amd();
    }
  );
}

extern "C" int __acpp_amdgpu_dpp_unsupported_on_rdna_or_non_amd();

template<int ctrl, int row_mask, int bank_mask, bool bound_ctrl>
inline int update_dpp(int val) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd &&
    jit::reflect<jit::reflection_query::target_arch>() < 0x1000,
    [&]() {
      return __acpp___builtin_amdgcn_update_dpp(
          0, val, ctrl, row_mask, bank_mask, bound_ctrl);
    },
    [&]() {
      return __acpp_amdgpu_dpp_unsupported_on_rdna_or_non_amd();
    }
  );
}

inline float unsafe_atomic_add_f32(float* ptr, float val) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd,
    [&]() {
      return __acpp___builtin_amdgcn_global_atomic_fadd_f32(
          (float __attribute__((address_space(1)))*)ptr, val);
    },
    [&]() {
      return static_cast<float>(__acpp_amdgpu_builtin_unsupported_on_non_amd());
    }
  );
}

inline double unsafe_atomic_add_f64(double* ptr, double val) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_vendor_id>() == jit::vendor_id::amd,
    [&]() {
      return __acpp___builtin_amdgcn_global_atomic_fadd_f64(
          (double __attribute__((address_space(1)))*)ptr, val);
    },
    [&]() {
      return static_cast<double>(__acpp_amdgpu_builtin_unsupported_on_non_amd());
    }
  );
}

inline float unsafe_atomic_fetch_add(hipsycl::sycl::access::address_space space,
                                     hipsycl::sycl::memory_order order,
                                     hipsycl::sycl::memory_scope scope,
                                     float* ptr, float val) {
  return unsafe_atomic_add_f32(ptr, val);
}

inline double unsafe_atomic_fetch_add(hipsycl::sycl::access::address_space space,
                                     hipsycl::sycl::memory_order order,
                                     hipsycl::sycl::memory_scope scope,
                                     double* ptr, double val) {
  return unsafe_atomic_add_f64(ptr, val);
}

} // namespace adaptivecpp::amdgpu

#endif // HIPSYCL_SSCP_AMDGPU_BUILTINS_HPP
