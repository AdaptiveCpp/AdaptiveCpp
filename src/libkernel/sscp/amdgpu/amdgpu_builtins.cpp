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

#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_builtins.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

extern "C" __acpp_int32 __acpp_sscp_custom_intrinsic__llvm_amdgcn_readfirstlane(
    __acpp_int32 input);

extern "C" __acpp_f32 __acpp_sscp_custom_intrinsic__llvm_amdgcn_fract_f32(
    __acpp_f32 input);

extern "C" float __acpp_sscp_custom_intrinsic__llvm_amdgcn_global_atomic_fadd_f32_p1_i32(
    const __attribute__((address_space(1))) float *, float);

extern "C" float __acpp_sscp_custom_intrinsic__llvm_amdgcn_flat_atomic_fadd_f32_p0_i32(
    float *, float);

extern "C" double __acpp_sscp_custom_intrinsic__llvm_amdgcn_global_atomic_fadd_f64_p1_i32(
    const __attribute__((address_space(1))) double *, double);

extern "C" double __acpp_sscp_custom_intrinsic__llvm_amdgcn_flat_atomic_fadd_f64_p0_i32(
    double *, double);

extern "C" float __acpp_sscp_custom_intrinsic__wrong_address_space_f32();
extern "C" double __acpp_sscp_custom_intrinsic__wrong_address_space_f64();

namespace adaptivecpp::amdgpu {

HIPSYCL_SSCP_BUILTIN int readfirstlane(int value) {
  return __acpp_sscp_custom_intrinsic__llvm_amdgcn_readfirstlane(value);

}

HIPSYCL_SSCP_BUILTIN float fract(float value) {
  return __acpp_sscp_custom_intrinsic__llvm_amdgcn_fract_f32(value);
}


HIPSYCL_SSCP_BUILTIN float unsafe_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, float *ptr, float x) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_arch>() == 0x90a,
    [&](){
      if (as == __acpp_sscp_address_space::global_space) {
        return __acpp_sscp_custom_intrinsic__llvm_amdgcn_global_atomic_fadd_f32_p1_i32(
            (const __attribute__((address_space(1))) float *)ptr, x);
      } else if (as == __acpp_sscp_address_space::generic_space) {
        return __acpp_sscp_custom_intrinsic__llvm_amdgcn_flat_atomic_fadd_f32_p0_i32(ptr, x);
      } else {
        return __acpp_sscp_atomic_fetch_add_f32(as, order, scope, ptr, x);
      }
    },
    [&]() {
      return __acpp_sscp_atomic_fetch_add_f32(as, order, scope, ptr, x);
    }
  );
}

HIPSYCL_SSCP_BUILTIN double unsafe_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, double *ptr, double x) {
  namespace jit = hipsycl::sycl::AdaptiveCpp_jit;
  return jit::compile_if_else(
    jit::reflect<jit::reflection_query::target_arch>() == 0x90a,
    [&](){
      if (as == __acpp_sscp_address_space::global_space) {
        return __acpp_sscp_custom_intrinsic__llvm_amdgcn_global_atomic_fadd_f64_p1_i32(
            (const __attribute__((address_space(1))) double *)ptr, x);
      } else if (as == __acpp_sscp_address_space::generic_space) {
        return __acpp_sscp_custom_intrinsic__llvm_amdgcn_flat_atomic_fadd_f64_p0_i32(ptr, x);
      } else {
        return __acpp_sscp_atomic_fetch_add_f64(as, order, scope, ptr, x);
      }
    },
    [&]() {
      return __acpp_sscp_atomic_fetch_add_f64(as, order, scope, ptr, x);
    }
  );
}


HIPSYCL_SSCP_BUILTIN_ATTRIBUTES float unsafe_atomic_fetch_add_impl(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, float *ptr, float x) {
  return unsafe_atomic_fetch_add_f32(as, order, scope, ptr, x);
}

HIPSYCL_SSCP_BUILTIN_ATTRIBUTES double unsafe_atomic_fetch_add_impl(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, double *ptr, double x) {
  return unsafe_atomic_fetch_add_f64(as, order, scope, ptr, x);
}



} // namespace adaptivecpp::amdgpu
