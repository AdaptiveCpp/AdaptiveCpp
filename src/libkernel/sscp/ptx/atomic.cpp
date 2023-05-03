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

#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"


// Atomic definitions adapted from __clang_cuda_device_functions.h

double __dAtomicAdd(double *__p, double __v) {
  return __nvvm_atom_add_gen_d(__p, __v);
}
double __dAtomicAdd_block(double *__p, double __v) {
  return __nvvm_atom_cta_add_gen_d(__p, __v);
}
double __dAtomicAdd_system(double *__p, double __v) {
  return __nvvm_atom_sys_add_gen_d(__p, __v);
}

float __fAtomicAdd(float *__p, float __v) {
  return __nvvm_atom_add_gen_f(__p, __v);
}
float __fAtomicAdd_block(float *__p, float __v) {
  return __nvvm_atom_cta_add_gen_f(__p, __v);
}
float __fAtomicAdd_system(float *__p, float __v) {
  return __nvvm_atom_sys_add_gen_f(__p, __v);
}
float __fAtomicExch(float *__p, float __v) {
  return __nv_int_as_float(
      __nvvm_atom_xchg_gen_i((int *)__p, __nv_float_as_int(__v)));
}
float __fAtomicExch_block(float *__p, float __v) {
  return __nv_int_as_float(
      __nvvm_atom_cta_xchg_gen_i((int *)__p, __nv_float_as_int(__v)));
}
float __fAtomicExch_system(float *__p, float __v) {
  return __nv_int_as_float(
      __nvvm_atom_sys_xchg_gen_i((int *)__p, __nv_float_as_int(__v)));
}

int __iAtomicAdd(int *__p, int __v) {
  return __nvvm_atom_add_gen_i(__p, __v);
}
int __iAtomicAdd_block(int *__p, int __v) {
  return __nvvm_atom_cta_add_gen_i(__p, __v);
}
int __iAtomicAdd_system(int *__p, int __v) {
  return __nvvm_atom_sys_add_gen_i(__p, __v);
}

int __iAtomicAnd(int *__p, int __v) {
  return __nvvm_atom_and_gen_i(__p, __v);
}
int __iAtomicAnd_block(int *__p, int __v) {
  return __nvvm_atom_cta_and_gen_i(__p, __v);
}
int __iAtomicAnd_system(int *__p, int __v) {
  return __nvvm_atom_sys_and_gen_i(__p, __v);
}

int __iAtomicCAS(int *__p, int __cmp, int __v) {
  return __nvvm_atom_cas_gen_i(__p, __cmp, __v);
}
int __iAtomicCAS_block(int *__p, int __cmp, int __v) {
  return __nvvm_atom_cta_cas_gen_i(__p, __cmp, __v);
}
int __iAtomicCAS_system(int *__p, int __cmp, int __v) {
  return __nvvm_atom_sys_cas_gen_i(__p, __cmp, __v);
}

int __iAtomicExch(int *__p, int __v) {
  return __nvvm_atom_xchg_gen_i(__p, __v);
}
int __iAtomicExch_block(int *__p, int __v) {
  return __nvvm_atom_cta_xchg_gen_i(__p, __v);
}
int __iAtomicExch_system(int *__p, int __v) {
  return __nvvm_atom_sys_xchg_gen_i(__p, __v);
}

int __iAtomicMax(int *__p, int __v) {
  return __nvvm_atom_max_gen_i(__p, __v);
}
int __iAtomicMax_block(int *__p, int __v) {
  return __nvvm_atom_cta_max_gen_i(__p, __v);
}
int __iAtomicMax_system(int *__p, int __v) {
  return __nvvm_atom_sys_max_gen_i(__p, __v);
}

int __iAtomicMin(int *__p, int __v) {
  return __nvvm_atom_min_gen_i(__p, __v);
}
int __iAtomicMin_block(int *__p, int __v) {
  return __nvvm_atom_cta_min_gen_i(__p, __v);
}
int __iAtomicMin_system(int *__p, int __v) {
  return __nvvm_atom_sys_min_gen_i(__p, __v);
}

int __iAtomicOr(int *__p, int __v) {
  return __nvvm_atom_or_gen_i(__p, __v);
}
int __iAtomicOr_block(int *__p, int __v) {
  return __nvvm_atom_cta_or_gen_i(__p, __v);
}
int __iAtomicOr_system(int *__p, int __v) {
  return __nvvm_atom_sys_or_gen_i(__p, __v);
}

int __iAtomicXor(int *__p, int __v) {
  return __nvvm_atom_xor_gen_i(__p, __v);
}
int __iAtomicXor_block(int *__p, int __v) {
  return __nvvm_atom_cta_xor_gen_i(__p, __v);
}
int __iAtomicXor_system(int *__p, int __v) {
  return __nvvm_atom_sys_xor_gen_i(__p, __v);
}

long long __illAtomicMax(long long *__p, long long __v) {
  return __nvvm_atom_max_gen_ll(__p, __v);
}
long long __illAtomicMax_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_max_gen_ll(__p, __v);
}
long long __illAtomicMax_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_max_gen_ll(__p, __v);
}

long long __illAtomicMin(long long *__p, long long __v) {
  return __nvvm_atom_min_gen_ll(__p, __v);
}
long long __illAtomicMin_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_min_gen_ll(__p, __v);
}
long long __illAtomicMin_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_min_gen_ll(__p, __v);
}

long long __llAtomicAnd(long long *__p, long long __v) {
  return __nvvm_atom_and_gen_ll(__p, __v);
}
long long __llAtomicAnd_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_and_gen_ll(__p, __v);
}
long long __llAtomicAnd_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_and_gen_ll(__p, __v);
}

long long __llAtomicOr(long long *__p, long long __v) {
  return __nvvm_atom_or_gen_ll(__p, __v);
}
long long __llAtomicOr_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_or_gen_ll(__p, __v);
}
long long __llAtomicOr_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_or_gen_ll(__p, __v);
}

long long __llAtomicXor(long long *__p, long long __v) {
  return __nvvm_atom_xor_gen_ll(__p, __v);
}
long long __llAtomicXor_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_xor_gen_ll(__p, __v);
}
long long __llAtomicXor_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_xor_gen_ll(__p, __v);
}

unsigned int __uAtomicAdd(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_add_gen_i((int *)__p, __v);
}
unsigned int __uAtomicAdd_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_add_gen_i((int *)__p, __v);
}
unsigned int __uAtomicAdd_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_add_gen_i((int *)__p, __v);
}

unsigned int __uAtomicAnd(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_and_gen_i((int *)__p, __v);
}
unsigned int __uAtomicAnd_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_and_gen_i((int *)__p, __v);
}
unsigned int __uAtomicAnd_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_and_gen_i((int *)__p, __v);
}

unsigned int __uAtomicCAS(unsigned int *__p, unsigned int __cmp,
                          unsigned int __v) {
  return __nvvm_atom_cas_gen_i((int *)__p, __cmp, __v);
}
unsigned int __uAtomicCAS_block(unsigned int *__p, unsigned int __cmp,
                                unsigned int __v) {
  return __nvvm_atom_cta_cas_gen_i((int *)__p, __cmp, __v);
}
unsigned int __uAtomicCAS_system(unsigned int *__p, unsigned int __cmp,
                                 unsigned int __v) {
  return __nvvm_atom_sys_cas_gen_i((int *)__p, __cmp, __v);
}

unsigned int __uAtomicDec(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_dec_gen_ui(__p, __v);
}
unsigned int __uAtomicDec_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_dec_gen_ui(__p, __v);
}
unsigned int __uAtomicDec_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_dec_gen_ui(__p, __v);
}

unsigned int __uAtomicExch(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_xchg_gen_i((int *)__p, __v);
}
unsigned int __uAtomicExch_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_xchg_gen_i((int *)__p, __v);
}
unsigned int __uAtomicExch_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_xchg_gen_i((int *)__p, __v);
}

unsigned int __uAtomicInc(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_inc_gen_ui(__p, __v);
}
unsigned int __uAtomicInc_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_inc_gen_ui(__p, __v);
}
unsigned int __uAtomicInc_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_inc_gen_ui(__p, __v);
}

unsigned int __uAtomicMax(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_max_gen_ui(__p, __v);
}
unsigned int __uAtomicMax_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_max_gen_ui(__p, __v);
}
unsigned int __uAtomicMax_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_max_gen_ui(__p, __v);
}

unsigned int __uAtomicMin(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_min_gen_ui(__p, __v);
}
unsigned int __uAtomicMin_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_min_gen_ui(__p, __v);
}
unsigned int __uAtomicMin_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_min_gen_ui(__p, __v);
}

unsigned int __uAtomicOr(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_or_gen_i((int *)__p, __v);
}
unsigned int __uAtomicOr_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_or_gen_i((int *)__p, __v);
}
unsigned int __uAtomicOr_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_or_gen_i((int *)__p, __v);
}

unsigned int __uAtomicXor(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_xor_gen_i((int *)__p, __v);
}
unsigned int __uAtomicXor_block(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_cta_xor_gen_i((int *)__p, __v);
}
unsigned int __uAtomicXor_system(unsigned int *__p, unsigned int __v) {
  return __nvvm_atom_sys_xor_gen_i((int *)__p, __v);
}

unsigned long long __ullAtomicAdd(unsigned long long *__p,
                                  unsigned long long __v) {
  return __nvvm_atom_add_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicAdd_block(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_add_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicAdd_system(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_add_gen_ll((long long *)__p, __v);
}

long long __llAtomicAdd(long long *__p, long long __v) {
  return __nvvm_atom_add_gen_ll(__p, __v);
}
long long __llAtomicAdd_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_add_gen_ll(__p, __v);
}
long long __llAtomicAdd_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_add_gen_ll(__p, __v);
}

unsigned long long __ullAtomicAnd(unsigned long long *__p,
                                  unsigned long long __v) {
  return __nvvm_atom_and_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicAnd_block(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_and_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicAnd_system(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_and_gen_ll((long long *)__p, __v);
}

long long __llAtomicCAS(long long *__p, long long __cmp, long long __v) {
  return __nvvm_atom_cas_gen_ll(__p, __cmp, __v);
}
long long __llAtomicCAS_block(long long *__p, long long __cmp, long long __v) {
  return __nvvm_atom_cta_cas_gen_ll(__p, __cmp, __v);
}
long long __llAtomicCAS_system(long long *__p, long long __cmp, long long __v) {
  return __nvvm_atom_sys_cas_gen_ll(__p, __cmp, __v);
}

unsigned long long __ullAtomicCAS(unsigned long long *__p,
                                  unsigned long long __cmp,
                                  unsigned long long __v) {
  return __nvvm_atom_cas_gen_ll((long long *)__p, __cmp, __v);
}
unsigned long long __ullAtomicCAS_block(unsigned long long *__p,
                                        unsigned long long __cmp,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_cas_gen_ll((long long *)__p, __cmp, __v);
}
unsigned long long __ullAtomicCAS_system(unsigned long long *__p,
                                         unsigned long long __cmp,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_cas_gen_ll((long long *)__p, __cmp, __v);
}

unsigned long long __ullAtomicExch(unsigned long long *__p,
                                   unsigned long long __v) {
  return __nvvm_atom_xchg_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicExch_block(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_cta_xchg_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicExch_system(unsigned long long *__p,
                                          unsigned long long __v) {
  return __nvvm_atom_sys_xchg_gen_ll((long long *)__p, __v);
}

long long __llAtomicExch(long long *__p, long long __v) {
  return __nvvm_atom_xchg_gen_ll(__p, __v);
}
long long __llAtomicExch_block(long long *__p, long long __v) {
  return __nvvm_atom_cta_xchg_gen_ll(__p, __v);
}
long long __llAtomicExch_system(long long *__p, long long __v) {
  return __nvvm_atom_sys_xchg_gen_ll(__p, __v);
}

unsigned long long __ullAtomicMax(unsigned long long *__p,
                                  unsigned long long __v) {
  return __nvvm_atom_max_gen_ull(__p, __v);
}
unsigned long long __ullAtomicMax_block(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_max_gen_ull(__p, __v);
}
unsigned long long __ullAtomicMax_system(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_max_gen_ull(__p, __v);
}

unsigned long long __ullAtomicMin(unsigned long long *__p,
                                  unsigned long long __v) {
  return __nvvm_atom_min_gen_ull(__p, __v);
}
unsigned long long __ullAtomicMin_block(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_min_gen_ull(__p, __v);
}
unsigned long long __ullAtomicMin_system(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_min_gen_ull(__p, __v);
}

unsigned long long __ullAtomicOr(unsigned long long *__p,
                                 unsigned long long __v) {
  return __nvvm_atom_or_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicOr_block(unsigned long long *__p,
                                       unsigned long long __v) {
  return __nvvm_atom_cta_or_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicOr_system(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_sys_or_gen_ll((long long *)__p, __v);
}

unsigned long long __ullAtomicXor(unsigned long long *__p,
                                  unsigned long long __v) {
  return __nvvm_atom_xor_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicXor_block(unsigned long long *__p,
                                        unsigned long long __v) {
  return __nvvm_atom_cta_xor_gen_ll((long long *)__p, __v);
}
unsigned long long __ullAtomicXor_system(unsigned long long *__p,
                                         unsigned long long __v) {
  return __nvvm_atom_sys_xor_gen_ll((long long *)__p, __v);
}





// ********************** atomic store ***************************

// Unlike the CUDA compilation flow, the __atomic_store and __atomic_load builtin
// generate incorrect code here. clang CUDA does not reall support them anyway, so
// for lack of a better alternative, implement them using trivial loads and stores and memfence.
// TODO: To what extent is this okay in the CUDA memory model?

void mem_fence(__hipsycl_sscp_memory_scope fence_scope) {
  if(fence_scope == hipsycl::sycl::memory_scope::system) {
    __nvvm_membar_sys();
  } else if(fence_scope == hipsycl::sycl::memory_scope::device) {
    __nvvm_membar_gl();
  } else if(fence_scope == hipsycl::sycl::memory_scope::work_group) {
    __nvvm_membar_cta();
  }
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  *ptr = x;
  mem_fence(scope);
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  *ptr = x;
  mem_fence(scope);
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  *ptr = x;
  mem_fence(scope);
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  *ptr = x;
  mem_fence(scope);
}


// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_load_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr) {
  return *ptr;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_load_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr) {
  return *ptr;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_load_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr) {
  return *ptr;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_load_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr) {
  return *ptr;
}

// for internal use only, not part of the public API
HIPSYCL_SSCP_BUILTIN __hipsycl_f32 atomic_load_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr) {
  return __nv_int_as_float(__hipsycl_sscp_atomic_load_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 atomic_load_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr) {
  return __nv_longlong_as_double(__hipsycl_sscp_atomic_load_i64(
      as, order, scope, reinterpret_cast<__hipsycl_int64 *>(ptr)));
}

// ********************** atomic exchange ***************************

HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_exchange_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  
  // We can only do 32-bit atomics, so we just treat it as a 32-bit value.
  // This is very bogus, but since sycl::atomic_ref does not support
  // types < 32bit, it's not user-facing anyway.
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_exchange_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_exchange_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr,
    __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_exchange_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_exchange_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr,
    __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicExch_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicExch(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicExch_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_exchange_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr,
    __hipsycl_int64 x) {
   if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicExch_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicExch(ptr, x);
  } else /* work group, sub group or work item */ {
    return __llAtomicExch_block(ptr, x);
  }
}

// ********************** atomic compare exchange weak **********************

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_weak_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int8 *ptr, __hipsycl_int8 *expected, __hipsycl_int8 desired) {
  return __hipsycl_sscp_cmp_exch_strong_i8(as, success, failure, scope, ptr,
                                           expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_weak_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int16 *ptr, __hipsycl_int16 *expected, __hipsycl_int16 desired) {
  return __hipsycl_sscp_cmp_exch_strong_i16(as, success, failure, scope, ptr,
                                            expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_weak_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int32 *ptr, __hipsycl_int32 *expected, __hipsycl_int32 desired) {
  return __hipsycl_sscp_cmp_exch_strong_i32(as, success, failure, scope, ptr,
                                            expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_weak_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int64 *ptr, __hipsycl_int64 *expected, __hipsycl_int64 desired) {
  return __hipsycl_sscp_cmp_exch_strong_i64(as, success, failure, scope, ptr,
                                            expected, desired);
}

// ********************* atomic compare exchange strong  *********************

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int8 *ptr, __hipsycl_int8 *expected, __hipsycl_int8 desired) {

  return __hipsycl_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      reinterpret_cast<__hipsycl_int32 *>(expected),
      static_cast<__hipsycl_int32>(desired));
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int16 *ptr, __hipsycl_int16 *expected, __hipsycl_int16 desired) {

  return __hipsycl_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      reinterpret_cast<__hipsycl_int32 *>(expected),
      static_cast<__hipsycl_int32>(desired));
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int32 *ptr, __hipsycl_int32 *expected, __hipsycl_int32 desired) {

  __hipsycl_int32 old = *expected;
  if (scope == __hipsycl_sscp_memory_scope::system) {
    *expected = __iAtomicCAS_system(ptr, *expected, desired);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    *expected = __iAtomicCAS(ptr, *expected, desired);
  } else /* work group, sub group or work item */ {
    *expected = __iAtomicCAS_block(ptr, *expected, desired);
  }
  return old == *expected;
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int64 *ptr, __hipsycl_int64 *expected, __hipsycl_int64 desired) {

  __hipsycl_int64 old = *expected;
  if (scope == __hipsycl_sscp_memory_scope::system) {
    *expected = __llAtomicCAS_system(ptr, *expected, desired);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    *expected = __llAtomicCAS(ptr, *expected, desired);
  } else /* work group, sub group or work item */ {
    *expected = __llAtomicCAS_block(ptr, *expected, desired);
  }
  return old == *expected;
}


// Only for internal use; not part of the public API
HIPSYCL_SSCP_BUILTIN bool cmp_exch_strong_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_f32 *ptr, __hipsycl_f32 *expected, __hipsycl_f32 desired) {

  return __hipsycl_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      reinterpret_cast<__hipsycl_int32 *>(expected),
      __nv_float_as_int(desired));
}

HIPSYCL_SSCP_BUILTIN bool cmp_exch_strong_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_f64 *ptr, __hipsycl_f64 *expected, __hipsycl_f64 desired) {

  return __hipsycl_sscp_cmp_exch_strong_i64(
      as, success, failure, scope, reinterpret_cast<__hipsycl_int64 *>(ptr),
      reinterpret_cast<__hipsycl_int64 *>(expected),
      __nv_double_as_longlong(desired));
}

// ******************** atomic fetch_and ************************

HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_and_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {

  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_and_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_and_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {

  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_and_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_and_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {

  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicAnd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicAnd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicAnd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_and_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {

  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicAnd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicAnd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __llAtomicAnd_block(ptr, x);
  }
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_or_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_or_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_or_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_or_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_or_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicOr_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicOr(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicOr_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_or_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicOr_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicOr(ptr, x);
  } else /* work group, sub group or work item */ {
    return __llAtomicOr_block(ptr, x);
  }
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_xor_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_xor_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_xor_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_xor_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_xor_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicXor_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicXor(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicXor_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_xor_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicXor_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicXor(ptr, x);
  } else /* work group, sub group or work item */ {
    return __llAtomicXor_block(ptr, x);
  }
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_add_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_add_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_add_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_add_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_add_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicAdd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_add_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __llAtomicAdd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_add_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return static_cast<__hipsycl_uint8>(__hipsycl_sscp_atomic_fetch_add_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_add_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return static_cast<__hipsycl_uint16>(__hipsycl_sscp_atomic_fetch_add_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_add_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __uAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __uAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __uAtomicAdd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_add_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __ullAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __ullAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __ullAtomicAdd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_add_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __fAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __fAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __fAtomicAdd_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_add_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __dAtomicAdd_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __dAtomicAdd(ptr, x);
  } else /* work group, sub group or work item */ {
    return __dAtomicAdd_block(ptr, x);
  }
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_sub_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_sub_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_sub_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_sub_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_sub_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicAdd_system(ptr, -x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicAdd(ptr, -x);
  } else /* work group, sub group or work item */ {
    return __iAtomicAdd_block(ptr, -x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_sub_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __llAtomicAdd_system(ptr, -x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __llAtomicAdd(ptr, -x);
  } else /* work group, sub group or work item */ {
    return __llAtomicAdd_block(ptr, -x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_sub_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return static_cast<__hipsycl_uint8>(__hipsycl_sscp_atomic_fetch_sub_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_sub_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return static_cast<__hipsycl_uint16>(__hipsycl_sscp_atomic_fetch_sub_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_sub_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __uAtomicAdd_system(ptr, (__hipsycl_uint32)-(int)x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __uAtomicAdd(ptr, (__hipsycl_uint32)-(int)x);
  } else /* work group, sub group or work item */ {
    return __uAtomicAdd_block(ptr, (__hipsycl_uint32)-(int)x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_sub_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __ullAtomicAdd_system(ptr, (__hipsycl_uint64)-(__hipsycl_int64)x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __ullAtomicAdd(ptr, (__hipsycl_uint64)-(__hipsycl_int64)x);
  } else /* work group, sub group or work item */ {
    return __ullAtomicAdd_block(ptr, (__hipsycl_uint64)-(__hipsycl_int64)x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_sub_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __fAtomicAdd_system(ptr, -x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __fAtomicAdd(ptr, -x);
  } else /* work group, sub group or work item */ {
    return __fAtomicAdd_block(ptr, -x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_sub_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __dAtomicAdd_system(ptr, -x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __dAtomicAdd(ptr, -x);
  } else /* work group, sub group or work item */ {
    return __dAtomicAdd_block(ptr, -x);
  }
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_min_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_min_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_min_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_min_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_min_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicMin_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicMin(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicMin_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_min_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __illAtomicMin_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __illAtomicMin(ptr, x);
  } else /* work group, sub group or work item */ {
    return __illAtomicMin_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_min_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return static_cast<__hipsycl_uint8>(__hipsycl_sscp_atomic_fetch_min_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_min_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return static_cast<__hipsycl_uint16>(__hipsycl_sscp_atomic_fetch_min_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_min_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __uAtomicMin_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __uAtomicMin(ptr, x);
  } else /* work group, sub group or work item */ {
    return __uAtomicMin_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_min_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __ullAtomicMin_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __ullAtomicMin(ptr, x);
  } else /* work group, sub group or work item */ {
    return __ullAtomicMin_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_min_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {

  __hipsycl_f32 old = atomic_load_f32(as, order, scope, ptr);

  do {
    if (old < x)
      return old;
  } while (!cmp_exch_strong_f32(as, order, order, scope, ptr, &old, x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_min_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  
  __hipsycl_f64 old = atomic_load_f64(as, order, scope, ptr);
  
  do {
    if (old < x)
      return old;
  } while (!cmp_exch_strong_f64(as, order, order, scope, ptr, &old, x));
  return x;
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_max_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return static_cast<__hipsycl_int8>(__hipsycl_sscp_atomic_fetch_max_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_max_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return static_cast<__hipsycl_int16>(__hipsycl_sscp_atomic_fetch_max_i32(
      as, order, scope, reinterpret_cast<__hipsycl_int32 *>(ptr),
      static_cast<__hipsycl_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_max_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __iAtomicMax_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __iAtomicMax(ptr, x);
  } else /* work group, sub group or work item */ {
    return __iAtomicMax_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_max_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __illAtomicMax_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __illAtomicMax(ptr, x);
  } else /* work group, sub group or work item */ {
    return __illAtomicMax_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_max_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return static_cast<__hipsycl_uint8>(__hipsycl_sscp_atomic_fetch_max_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_max_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return static_cast<__hipsycl_uint16>(__hipsycl_sscp_atomic_fetch_max_u32(
      as, order, scope, reinterpret_cast<__hipsycl_uint32 *>(ptr),
      static_cast<__hipsycl_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_max_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __uAtomicMax_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __uAtomicMax(ptr, x);
  } else /* work group, sub group or work item */ {
    return __uAtomicMax_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_max_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  if (scope == __hipsycl_sscp_memory_scope::system) {
    return __ullAtomicMax_system(ptr, x);
  } else if (scope == __hipsycl_sscp_memory_scope::device) {
    return __ullAtomicMax(ptr, x);
  } else /* work group, sub group or work item */ {
    return __ullAtomicMax_block(ptr, x);
  }
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_max_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  
  __hipsycl_f32 old = atomic_load_f32(as, order, scope, ptr);
  
  do {
    if (old > x)
      return old;
  } while (!cmp_exch_strong_f32(as, order, order, scope, ptr, &old, x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_max_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  
  __hipsycl_f64 old = atomic_load_f64(as, order, scope, ptr);
  
  do {
    if (old > x)
      return old;
  } while (!cmp_exch_strong_f64(as, order, order, scope, ptr, &old, x));
  return x;
}


