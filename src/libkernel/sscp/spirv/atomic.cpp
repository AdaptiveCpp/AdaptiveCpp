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
#include "hipSYCL/sycl/libkernel/sscp/builtins/spirv/spirv_common.hpp"

template<class T>
__spirv_global T* to_global(T* ptr) { return (__spirv_global T*)ptr; }

template<class T>
__spirv_local T* to_local(T* ptr) { return (__spirv_local T*)ptr; }

#define SPIRV_DECLARE_ATOMIC_LOAD(AS, T)                                       \
  T __spirv_AtomicLoad(AS T *ptr, __spv::ScopeFlag scope,                      \
                       __spv::MemorySemanticsMaskFlag semantics);

#define SPIRV_DECLARE_ATOMIC_STORE(AS, T)                                      \
  void __spirv_AtomicStore(AS T *ptr, __spv::ScopeFlag scope,                  \
                           __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_EXCHANGE(AS, T)                                   \
  T __spirv_AtomicExchange(AS T *ptr, __spv::ScopeFlag S,                      \
                           __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_AND(AS, T)                                        \
  T __spirv_AtomicAnd(AS T *ptr, __spv::ScopeFlag scope,                       \
                      __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_OR(AS, T)                                         \
  T __spirv_AtomicOr(AS T *ptr, __spv::ScopeFlag scope,                        \
                     __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_XOR(AS, T)                                        \
  T __spirv_AtomicXor(AS T *ptr, __spv::ScopeFlag scope,                       \
                      __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_COMPARE_EXCHANGE(AS, T)                           \
  T __spirv_AtomicCompareExchange(                                             \
      AS T *ptr, __spv::ScopeFlag S, __spv::MemorySemanticsMaskFlag success,   \
      __spv::MemorySemanticsMaskFlag failure, T desired, T expected);

#define SPIRV_DECLARE_ATOMIC_IADD(AS, T)                                       \
  T __spirv_AtomicIAdd(AS T *ptr, __spv::ScopeFlag scope,                      \
                      __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_FADD(AS, T)                                       \
  T __spirv_AtomicFAddEXT(AS T *ptr, __spv::ScopeFlag scope,                   \
                          __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_ISUB(AS, T)                                       \
  T __spirv_AtomicISub(AS T *ptr, __spv::ScopeFlag scope,                      \
                      __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_SMIN(AS, T)                                       \
  T __spirv_AtomicSMin(AS T *ptr, __spv::ScopeFlag scope,                      \
                       __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_UMIN(AS, T)                                       \
  T __spirv_AtomicUMin(AS T *ptr, __spv::ScopeFlag scope,                      \
                       __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_FMIN(AS, T)                                       \
  T __spirv_AtomicFMinEXT(AS T *ptr, __spv::ScopeFlag scope,                   \
                          __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_SMAX(AS, T)                                       \
  T __spirv_AtomicSMax(AS T *ptr, __spv::ScopeFlag scope,                      \
                       __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_UMAX(AS, T)                                       \
  T __spirv_AtomicUMax(AS T *ptr, __spv::ScopeFlag scope,                      \
                       __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMIC_FMAX(AS, T)                                       \
  T __spirv_AtomicFMaxEXT(AS T *ptr, __spv::ScopeFlag scope,                   \
                          __spv::MemorySemanticsMaskFlag semantics, T x);

#define SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, Type)              \
  Generator(__spirv_global, Type);                                             \
  Generator(__spirv_local, Type);                                              \
  Generator(__spirv_generic, Type);

#define SPIRV_DECLARE_ATOMICS_FOR_SIGNED(Generator)                            \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_int8);         \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_int16);        \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_int32);        \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_int64);

#define SPIRV_DECLARE_ATOMICS_FOR_UNSIGNED(Generator)                          \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_uint8);        \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_uint16);       \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_uint32);       \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_uint64);

#define SPIRV_DECLARE_ATOMICS_FOR_FLOAT(Generator)                             \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_f32);          \
  SPIRV_DECLARE_ATOMICS_FOR_ADDRESS_SPACES(Generator, __acpp_f64);

SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_LOAD)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_STORE)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_EXCHANGE)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_COMPARE_EXCHANGE)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_AND)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_OR)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_XOR)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_IADD)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_ISUB)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_SMIN)
SPIRV_DECLARE_ATOMICS_FOR_SIGNED(SPIRV_DECLARE_ATOMIC_SMAX)

SPIRV_DECLARE_ATOMICS_FOR_UNSIGNED(SPIRV_DECLARE_ATOMIC_IADD)
SPIRV_DECLARE_ATOMICS_FOR_UNSIGNED(SPIRV_DECLARE_ATOMIC_ISUB)
SPIRV_DECLARE_ATOMICS_FOR_UNSIGNED(SPIRV_DECLARE_ATOMIC_UMIN)
SPIRV_DECLARE_ATOMICS_FOR_UNSIGNED(SPIRV_DECLARE_ATOMIC_UMAX)

SPIRV_DECLARE_ATOMICS_FOR_FLOAT(SPIRV_DECLARE_ATOMIC_FADD)
SPIRV_DECLARE_ATOMICS_FOR_FLOAT(SPIRV_DECLARE_ATOMIC_FMIN)
SPIRV_DECLARE_ATOMICS_FOR_FLOAT(SPIRV_DECLARE_ATOMIC_FMAX)

__attribute__((always_inline)) __spv::MemorySemanticsMaskFlag
get_atomic_memory_semantics(__acpp_sscp_memory_order order) {
  __acpp_uint32 semantics_from_order = get_spirv_memory_semantics(order);

  return static_cast<__spv::MemorySemanticsMaskFlag>(
      semantics_from_order | __spv::MemorySemanticsMaskFlag::SubgroupMemory |
      __spv::MemorySemanticsMaskFlag::WorkgroupMemory |
      __spv::MemorySemanticsMaskFlag::CrossWorkgroupMemory);
}

#define ADDRESS_SPACE_SWITCH(as, ptr, command)                                 \
  if (as == __acpp_sscp_address_space::global_space) {                      \
    command(to_global(ptr));                                                   \
  } else if (as == __acpp_sscp_address_space::local_space) {                \
    command(to_local(ptr));                                                    \
  } else {                                                                     \
    command(ptr);                                                              \
  }

// ********************** atomic store ***************************

#define INVOKE_ATOMIC_STORE(ptr)                                               \
  __spirv_AtomicStore(ptr, get_spirv_scope(scope),                             \
                      get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, INVOKE_ATOMIC_STORE);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, INVOKE_ATOMIC_STORE);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, INVOKE_ATOMIC_STORE);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, INVOKE_ATOMIC_STORE);
}



// ********************** atomic load ***************************

#define RETURN_ATOMIC_LOAD(ptr)                                                \
  return __spirv_AtomicLoad(ptr, get_spirv_scope(scope),                       \
                            get_atomic_memory_semantics(order));

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_load_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_LOAD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_load_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_LOAD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_LOAD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_load_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_LOAD);
}



// ********************** atomic exchange ***************************

#define RETURN_ATOMIC_EXCHANGE(ptr)                                            \
  return __spirv_AtomicExchange(ptr, get_spirv_scope(scope),                   \
                                get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_exchange_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_exchange_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr,
    __acpp_int16 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_exchange_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr,
    __acpp_int32 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_exchange_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr,
    __acpp_int64 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_EXCHANGE);
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

#define RETURN_ATOMIC_COMPARE_EXCHANGE(ptr)                                    \
  auto old = *expected;                                                        \
  *expected = __spirv_AtomicCompareExchange(                                   \
      ptr, get_spirv_scope(scope), get_atomic_memory_semantics(success),       \
      get_atomic_memory_semantics(failure), desired, *expected);               \
  return old == *expected;

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int8 *ptr, __acpp_int8 *expected, __acpp_int8 desired) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_COMPARE_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int16 *ptr, __acpp_int16 *expected, __acpp_int16 desired) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_COMPARE_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_COMPARE_EXCHANGE);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_COMPARE_EXCHANGE);
}

// ************************** bitwise atomic operations *******************

#define RETURN_ATOMIC_AND(ptr)                                                 \
  return __spirv_AtomicAnd(ptr, get_spirv_scope(scope),                        \
                           get_atomic_memory_semantics(order), x);


HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_and_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_AND);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_and_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_AND);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_and_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_AND);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_and_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {

  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_AND);
}

#define RETURN_ATOMIC_OR(ptr)                                                  \
  return __spirv_AtomicOr(ptr, get_spirv_scope(scope),                         \
                          get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_or_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_OR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_or_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_OR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_or_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_OR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_or_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_OR);
}

#define RETURN_ATOMIC_XOR(ptr)                                                 \
  return __spirv_AtomicXor(ptr, get_spirv_scope(scope),                        \
                           get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_xor_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_XOR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_xor_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_XOR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_xor_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_XOR);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_xor_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_XOR);
}




#define RETURN_ATOMIC_IADD(ptr)                                                 \
  return __spirv_AtomicIAdd(ptr, get_spirv_scope(scope),                        \
                           get_atomic_memory_semantics(order), x);
#define RETURN_ATOMIC_FADD(ptr)                                                \
  return __spirv_AtomicFAddEXT(ptr, get_spirv_scope(scope),                    \
                               get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_add_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_add_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_add_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_add_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_add_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_add_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_add_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_add_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_IADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FADD);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FADD);
}


#define RETURN_ATOMIC_ISUB(ptr)                                                \
  return __spirv_AtomicISub(ptr, get_spirv_scope(scope),                       \
                           get_atomic_memory_semantics(order), x);
#define RETURN_ATOMIC_FSUB(ptr)                                                \
  return __spirv_AtomicFAddEXT(ptr, get_spirv_scope(scope),                    \
                               get_atomic_memory_semantics(order), -x);


HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_sub_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_sub_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_sub_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_sub_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_sub_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_sub_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_sub_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_sub_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_ISUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_sub_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FSUB);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_sub_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FSUB);
}

#define RETURN_ATOMIC_SMIN(ptr)                                                \
  return __spirv_AtomicSMin(ptr, get_spirv_scope(scope),                       \
                            get_atomic_memory_semantics(order), x);

#define RETURN_ATOMIC_UMIN(ptr)                                                \
  return __spirv_AtomicUMin(ptr, get_spirv_scope(scope),                       \
                            get_atomic_memory_semantics(order), x);

#define RETURN_ATOMIC_FMIN(ptr)                                                \
  return __spirv_AtomicFMinEXT(ptr, get_spirv_scope(scope),                    \
                               get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_min_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_min_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_min_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_min_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_min_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_min_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_min_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_min_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_min_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FMIN);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_min_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FMIN);
}

#define RETURN_ATOMIC_SMAX(ptr)                                                \
  return __spirv_AtomicSMax(ptr, get_spirv_scope(scope),                       \
                            get_atomic_memory_semantics(order), x);

#define RETURN_ATOMIC_UMAX(ptr)                                                \
  return __spirv_AtomicUMax(ptr, get_spirv_scope(scope),                       \
                            get_atomic_memory_semantics(order), x);

#define RETURN_ATOMIC_FMAX(ptr)                                                \
  return __spirv_AtomicFMaxEXT(ptr, get_spirv_scope(scope),                    \
                               get_atomic_memory_semantics(order), x);

HIPSYCL_SSCP_BUILTIN __acpp_int8 __acpp_sscp_atomic_fetch_max_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int8 *ptr, __acpp_int8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_int16 __acpp_sscp_atomic_fetch_max_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int16 *ptr, __acpp_int16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_max_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_max_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_SMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_atomic_fetch_max_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint8 *ptr, __acpp_uint8 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_atomic_fetch_max_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint16 *ptr, __acpp_uint16 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_max_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_max_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_UMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_max_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FMAX);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_max_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  
  ADDRESS_SPACE_SWITCH(as, ptr, RETURN_ATOMIC_FMAX);
}

