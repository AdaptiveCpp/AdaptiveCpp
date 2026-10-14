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
#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/ptx/libdevice.hpp"


extern "C" int __acpp_sscp_jit_reflect_target_arch();

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

void mem_fence(__acpp_sscp_memory_scope fence_scope) {
  if(fence_scope == hipsycl::sycl::memory_scope::system) {
    __nvvm_membar_sys();
  } else if(fence_scope == hipsycl::sycl::memory_scope::device) {
    __nvvm_membar_gl();
  } else {
    // Atomic operations do not accept work-item scope. Using a CTA fence for
    // sub-group scope is conservative and therefore valid.
    __nvvm_membar_cta();
  }
}

void atomic_rmw_pre_fence(__acpp_sscp_memory_order order,
                          __acpp_sscp_memory_scope scope) {
  if(order == __acpp_sscp_memory_order::release ||
     order == __acpp_sscp_memory_order::acq_rel ||
     order == __acpp_sscp_memory_order::seq_cst)
    mem_fence(scope);
}

void atomic_rmw_post_fence(__acpp_sscp_memory_order order,
                           __acpp_sscp_memory_scope scope) {
  if(order == __acpp_sscp_memory_order::acquire ||
     order == __acpp_sscp_memory_order::acq_rel ||
     order == __acpp_sscp_memory_order::seq_cst)
    mem_fence(scope);
}

__acpp_sscp_memory_order
load_order_for_rmw(__acpp_sscp_memory_order order) {
  if(order == __acpp_sscp_memory_order::release)
    return __acpp_sscp_memory_order::relaxed;
  if(order == __acpp_sscp_memory_order::acq_rel)
    return __acpp_sscp_memory_order::acquire;
  return order;
}

__acpp_sscp_memory_order
failure_order_for_rmw(__acpp_sscp_memory_order order) {
  if(order == __acpp_sscp_memory_order::release)
    return __acpp_sscp_memory_order::relaxed;
  if(order == __acpp_sscp_memory_order::acq_rel)
    return __acpp_sscp_memory_order::acquire;
  return order;
}

#define ACPP_PTX_ORDERED_RMW_RETURN(expr)                                     \
  do {                                                                         \
    atomic_rmw_pre_fence(order, scope);                                        \
    const auto __acpp_result = (expr);                                         \
    atomic_rmw_post_fence(order, scope);                                       \
    return __acpp_result;                                                      \
  } while(false)


template<class T>
T memfenced_load(T* ptr, __acpp_sscp_memory_scope scope) {
  mem_fence(scope);
  T x = *ptr;
  mem_fence(scope);
  return x;
}

template<class T>
void memfenced_store(T* ptr, T x, __acpp_sscp_memory_scope scope) {
  mem_fence(scope);
  *ptr = x;
  mem_fence(scope);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  memfenced_store(ptr, x, scope);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  memfenced_store(ptr, x, scope);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if(__acpp_sscp_jit_reflect_target_arch() >= 70) {
    if(scope == __acpp_sscp_memory_scope::system) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        asm volatile("st.relaxed.sys.s32 [%0], %1;"
                     :
                     : "l"(ptr), "r"(x)
                     : "memory");
        return;
      } else if(order == __acpp_sscp_memory_order::release) {
        asm volatile("st.release.sys.s32 [%0], %1;"
                     :
                     : "l"(ptr), "r"(x)
                     : "memory");
        return;
      }
    } else if(scope == __acpp_sscp_memory_scope::device) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        asm volatile("st.relaxed.gpu.s32 [%0], %1;"
                     :
                     : "l"(ptr), "r"(x)
                     : "memory");
        return;
      } else if(order == __acpp_sscp_memory_order::release) {
        asm volatile("st.release.gpu.s32 [%0], %1;"
                     :
                     : "l"(ptr), "r"(x)
                     : "memory");
        return;
      }
    }
  }

  atomic_rmw_pre_fence(order, scope);
  if(scope == __acpp_sscp_memory_scope::system)
    (void)__iAtomicExch_system(ptr, x);
  else if(scope == __acpp_sscp_memory_scope::device)
    (void)__iAtomicExch(ptr, x);
  else
    (void)__iAtomicExch_block(ptr, x);
  atomic_rmw_post_fence(order, scope);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if(__acpp_sscp_jit_reflect_target_arch() >= 70) {
    if(scope == __acpp_sscp_memory_scope::system) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        asm volatile("st.relaxed.sys.s64 [%0], %1;"
                     :
                     : "l"(ptr), "l"(x)
                     : "memory");
        return;
      } else if(order == __acpp_sscp_memory_order::release) {
        asm volatile("st.release.sys.s64 [%0], %1;"
                     :
                     : "l"(ptr), "l"(x)
                     : "memory");
        return;
      }
    } else if(scope == __acpp_sscp_memory_scope::device) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        asm volatile("st.relaxed.gpu.s64 [%0], %1;"
                     :
                     : "l"(ptr), "l"(x)
                     : "memory");
        return;
      } else if(order == __acpp_sscp_memory_order::release) {
        asm volatile("st.release.gpu.s64 [%0], %1;"
                     :
                     : "l"(ptr), "l"(x)
                     : "memory");
        return;
      }
    }
  }

  atomic_rmw_pre_fence(order, scope);
  if(scope == __acpp_sscp_memory_scope::system)
    (void)__llAtomicExch_system(ptr, x);
  else if(scope == __acpp_sscp_memory_scope::device)
    (void)__llAtomicExch(ptr, x);
  else
    (void)__llAtomicExch_block(ptr, x);
  atomic_rmw_post_fence(order, scope);
}


// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_load_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr) {
  return memfenced_load(ptr, scope);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_load_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr) {
  return memfenced_load(ptr, scope);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr) {
  if(__acpp_sscp_jit_reflect_target_arch() >= 70) {
    if(scope == __acpp_sscp_memory_scope::system) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        __acpp_int32 result;
        asm volatile("ld.relaxed.sys.u32 %0,[%1];"
                     : "=r"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      } else if(order == __acpp_sscp_memory_order::acquire) {
        __acpp_int32 result;
        asm volatile("ld.acquire.sys.u32 %0,[%1];"
                     : "=r"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      }
    } else if(scope == __acpp_sscp_memory_scope::device) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        __acpp_int32 result;
        asm volatile("ld.relaxed.gpu.u32 %0,[%1];"
                     : "=r"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      } else if(order == __acpp_sscp_memory_order::acquire) {
        __acpp_int32 result;
        asm volatile("ld.acquire.gpu.u32 %0,[%1];"
                     : "=r"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      }
    }
  }

  __acpp_int32 result;
  if(scope == __acpp_sscp_memory_scope::system)
    result = __iAtomicCAS_system(ptr, 0, 0);
  else if(scope == __acpp_sscp_memory_scope::device)
    result = __iAtomicCAS(ptr, 0, 0);
  else
    result = __iAtomicCAS_block(ptr, 0, 0);
  atomic_rmw_post_fence(order, scope);
  return result;
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_load_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr) {
  if(__acpp_sscp_jit_reflect_target_arch() >= 70) {
    if(scope == __acpp_sscp_memory_scope::system) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        __acpp_int64 result;
        asm volatile("ld.relaxed.sys.u64 %0,[%1];"
                     : "=l"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      } else if(order == __acpp_sscp_memory_order::acquire) {
        __acpp_int64 result;
        asm volatile("ld.acquire.sys.u64 %0,[%1];"
                     : "=l"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      }
    } else if(scope == __acpp_sscp_memory_scope::device) {
      if(order == __acpp_sscp_memory_order::relaxed) {
        __acpp_int64 result;
        asm volatile("ld.relaxed.gpu.u64 %0,[%1];"
                     : "=l"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      } else if(order == __acpp_sscp_memory_order::acquire) {
        __acpp_int64 result;
        asm volatile("ld.acquire.gpu.u64 %0,[%1];"
                     : "=l"(result)
                     : "l"(ptr)
                     : "memory");
        return result;
      }
    }
  }

  __acpp_int64 result;
  if(scope == __acpp_sscp_memory_scope::system)
    result = __llAtomicCAS_system(ptr, 0, 0);
  else if(scope == __acpp_sscp_memory_scope::device)
    result = __llAtomicCAS(ptr, 0, 0);
  else
    result = __llAtomicCAS_block(ptr, 0, 0);
  atomic_rmw_post_fence(order, scope);
  return result;
}

// for internal use only, not part of the public API
HIPSYCL_SSCP_BUILTIN __acpp_f32 atomic_load_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr) {
  return __nv_int_as_float(__acpp_sscp_atomic_load_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr)));
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 atomic_load_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr) {
  return __nv_longlong_as_double(__acpp_sscp_atomic_load_i64(
      as, order, scope, reinterpret_cast<__acpp_int64 *>(ptr)));
}

// ********************** atomic exchange ***************************

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_exchange_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  // We can only do 32-bit atomics, so we just treat it as a 32-bit value.
  // This is very bogus, but since sycl::atomic_ref does not support
  // types < 32bit, it's not user-facing anyway.
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_exchange_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_exchange_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr,
    __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_exchange_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_exchange_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr,
    __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicExch_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicExch(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicExch_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_exchange_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr,
    __acpp_int64 x) {
   if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicExch_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicExch(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicExch_block(ptr, x));
  }
}

// ********************** atomic compare exchange weak **********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int8 *ptr, __acpp_int8 *expected, __acpp_int8 desired) {
  return __acpp_sscp_cmp_exch_strong_i8(as, success, failure, scope, ptr,
                                           expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int16 *ptr, __acpp_int16 *expected, __acpp_int16 desired) {
  return __acpp_sscp_cmp_exch_strong_i16(as, success, failure, scope, ptr,
                                            expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {
  return __acpp_sscp_cmp_exch_strong_i32(as, success, failure, scope, ptr,
                                            expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {
  return __acpp_sscp_cmp_exch_strong_i64(as, success, failure, scope, ptr,
                                            expected, desired);
}

// ********************* atomic compare exchange strong  *********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int8 *ptr, __acpp_int8 *expected, __acpp_int8 desired) {

  return __acpp_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      reinterpret_cast<__acpp_int32 *>(expected),
      static_cast<__acpp_int32>(desired));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int16 *ptr, __acpp_int16 *expected, __acpp_int16 desired) {

  return __acpp_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      reinterpret_cast<__acpp_int32 *>(expected),
      static_cast<__acpp_int32>(desired));
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {
  atomic_rmw_pre_fence(success, scope);

  const __acpp_int32 old = *expected;
  if(scope == __acpp_sscp_memory_scope::system)
    *expected = __iAtomicCAS_system(ptr, *expected, desired);
  else if(scope == __acpp_sscp_memory_scope::device)
    *expected = __iAtomicCAS(ptr, *expected, desired);
  else
    *expected = __iAtomicCAS_block(ptr, *expected, desired);

  const bool succeeded = old == *expected;
  atomic_rmw_post_fence(succeeded ? success : failure, scope);
  return succeeded;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {
  atomic_rmw_pre_fence(success, scope);

  const __acpp_int64 old = *expected;
  if(scope == __acpp_sscp_memory_scope::system)
    *expected = __llAtomicCAS_system(ptr, *expected, desired);
  else if(scope == __acpp_sscp_memory_scope::device)
    *expected = __llAtomicCAS(ptr, *expected, desired);
  else
    *expected = __llAtomicCAS_block(ptr, *expected, desired);

  const bool succeeded = old == *expected;
  atomic_rmw_post_fence(succeeded ? success : failure, scope);
  return succeeded;
}


// Only for internal use; not part of the public API
HIPSYCL_SSCP_BUILTIN bool cmp_exch_strong_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_f32 *ptr, __acpp_f32 *expected, __acpp_f32 desired) {

  return __acpp_sscp_cmp_exch_strong_i32(
      as, success, failure, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      reinterpret_cast<__acpp_int32 *>(expected),
      __nv_float_as_int(desired));
}

HIPSYCL_SSCP_BUILTIN bool cmp_exch_strong_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_f64 *ptr, __acpp_f64 *expected, __acpp_f64 desired) {

  return __acpp_sscp_cmp_exch_strong_i64(
      as, success, failure, scope, reinterpret_cast<__acpp_int64 *>(ptr),
      reinterpret_cast<__acpp_int64 *>(expected),
      __nv_double_as_longlong(desired));
}

// ******************** atomic fetch_and ************************

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_and_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {

  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_and_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_and_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {

  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_and_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_and_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {

  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAnd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAnd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAnd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_and_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {

  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAnd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAnd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAnd_block(ptr, x));
  }
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_or_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_or_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_or_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_or_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_or_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicOr_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicOr(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicOr_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_or_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicOr_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicOr(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicOr_block(ptr, x));
  }
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_xor_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_xor_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_xor_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_xor_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_xor_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicXor_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicXor(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicXor_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_xor_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicXor_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicXor(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicXor_block(ptr, x));
  }
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_add_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_add_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_add_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_add_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_add_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  if (scope == __acpp_sscp_memory_scope::system) {
    if(order == __acpp_sscp_memory_order::acq_rel) {
      __acpp_int32 result;
      asm volatile("atom.add.acq_rel.sys.s32 %0,[%1],%2;"
                          : "=r"(result)
                          : "l"(ptr), "r"(x)
                          : "memory");
      return result;
    }
    else  
      ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_add_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_add_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  return static_cast<__acpp_uint8>(__acpp_sscp_atomic_fetch_add_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_add_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  return static_cast<__acpp_uint16>(__acpp_sscp_atomic_fetch_add_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_add_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    if(order == __acpp_sscp_memory_order::acq_rel) {
      __acpp_uint32 result;
      asm volatile("atom.add.acq_rel.sys.u32 %0,[%1],%2;"
                          : "=r"(result)
                          : "l"(ptr), "r"(x)
                          : "memory");
      return result;
    }
    else  
      ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_add_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd_block(ptr, x));
  }
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_sub_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_sub_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_sub_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_sub_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_sub_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd_system(ptr, -x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd(ptr, -x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicAdd_block(ptr, -x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_sub_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd_system(ptr, -x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd(ptr, -x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__llAtomicAdd_block(ptr, -x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_sub_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  return static_cast<__acpp_uint8>(__acpp_sscp_atomic_fetch_sub_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_sub_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  return static_cast<__acpp_uint16>(__acpp_sscp_atomic_fetch_sub_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_sub_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd_system(ptr, (__acpp_uint32)-(int)x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd(ptr, (__acpp_uint32)-(int)x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicAdd_block(ptr, (__acpp_uint32)-(int)x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_sub_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd_system(ptr, (__acpp_uint64)-(__acpp_int64)x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd(ptr, (__acpp_uint64)-(__acpp_int64)x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicAdd_block(ptr, (__acpp_uint64)-(__acpp_int64)x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_sub_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd_system(ptr, -x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd(ptr, -x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__fAtomicAdd_block(ptr, -x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_sub_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd_system(ptr, -x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd(ptr, -x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__dAtomicAdd_block(ptr, -x));
  }
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_min_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_min_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_min_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_min_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_min_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMin_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMin(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMin_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_min_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMin_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMin(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMin_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_min_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  return static_cast<__acpp_uint8>(__acpp_sscp_atomic_fetch_min_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_min_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  return static_cast<__acpp_uint16>(__acpp_sscp_atomic_fetch_min_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_min_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMin_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMin(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMin_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_min_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMin_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMin(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMin_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_min_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {

  __acpp_f32 old = atomic_load_f32(as, load_order_for_rmw(order), scope, ptr);

  do {
    if (old < x)
      return old;
  } while (!cmp_exch_strong_f32(as, order, failure_order_for_rmw(order), scope, ptr, &old, x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_min_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  
  __acpp_f64 old = atomic_load_f64(as, load_order_for_rmw(order), scope, ptr);
  
  do {
    if (old < x)
      return old;
  } while (!cmp_exch_strong_f64(as, order, failure_order_for_rmw(order), scope, ptr, &old, x));
  return x;
}



HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_max_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  return static_cast<__acpp_int8>(__acpp_sscp_atomic_fetch_max_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_max_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  return static_cast<__acpp_int16>(__acpp_sscp_atomic_fetch_max_i32(
      as, order, scope, reinterpret_cast<__acpp_int32 *>(ptr),
      static_cast<__acpp_int32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_max_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMax_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMax(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__iAtomicMax_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_max_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMax_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMax(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__illAtomicMax_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_max_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  return static_cast<__acpp_uint8>(__acpp_sscp_atomic_fetch_max_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_max_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  return static_cast<__acpp_uint16>(__acpp_sscp_atomic_fetch_max_u32(
      as, order, scope, reinterpret_cast<__acpp_uint32 *>(ptr),
      static_cast<__acpp_uint32>(x)));
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_max_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMax_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMax(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__uAtomicMax_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_max_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  if (scope == __acpp_sscp_memory_scope::system) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMax_system(ptr, x));
  } else if (scope == __acpp_sscp_memory_scope::device) {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMax(ptr, x));
  } else /* work group, sub group or work item */ {
    ACPP_PTX_ORDERED_RMW_RETURN(__ullAtomicMax_block(ptr, x));
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_max_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  
  __acpp_f32 old = atomic_load_f32(as, load_order_for_rmw(order), scope, ptr);
  
  do {
    if (old > x)
      return old;
  } while (!cmp_exch_strong_f32(as, order, failure_order_for_rmw(order), scope, ptr, &old, x));
  return x;
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_max_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  
  __acpp_f64 old = atomic_load_f64(as, load_order_for_rmw(order), scope, ptr);
  
  do {
    if (old > x)
      return old;
  } while (!cmp_exch_strong_f64(as, order, failure_order_for_rmw(order), scope, ptr, &old, x));
  return x;
}

#undef ACPP_PTX_ORDERED_RMW_RETURN
