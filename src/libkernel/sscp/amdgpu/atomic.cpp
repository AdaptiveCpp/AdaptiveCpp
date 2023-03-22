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


inline constexpr int builtin_memory_order(__hipsycl_sscp_memory_order o) noexcept {
  switch(o){
    case __hipsycl_sscp_memory_order::relaxed:
      return __ATOMIC_RELAXED;
    case __hipsycl_sscp_memory_order::acquire:
      return __ATOMIC_ACQUIRE;
    case __hipsycl_sscp_memory_order::release:
      return __ATOMIC_RELEASE;
    case __hipsycl_sscp_memory_order::acq_rel:
      return __ATOMIC_ACQ_REL;
    case __hipsycl_sscp_memory_order::seq_cst:
      return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_RELAXED;
}

#ifndef __HIP_MEMORY_SCOPE_SINGLETHREAD
 #define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#endif

#ifndef __HIP_MEMORY_SCOPE_WAVEFRONT
 #define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#endif

#ifndef __HIP_MEMORY_SCOPE_WORKGROUP
 #define __HIP_MEMORY_SCOPE_WORKGROUP 3
#endif

#ifndef __HIP_MEMORY_SCOPE_AGENT
 #define __HIP_MEMORY_SCOPE_AGENT 4
#endif

#ifndef __HIP_MEMORY_SCOPE_SYSTEM
 #define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

inline constexpr int builtin_memory_scope(__hipsycl_sscp_memory_scope s) noexcept {
  switch(s) {
    case __hipsycl_sscp_memory_scope::work_item:
      return __HIP_MEMORY_SCOPE_SINGLETHREAD;
    case __hipsycl_sscp_memory_scope::sub_group:
      return __HIP_MEMORY_SCOPE_WAVEFRONT;
    case __hipsycl_sscp_memory_scope::work_group:
      return __HIP_MEMORY_SCOPE_WORKGROUP;
    case __hipsycl_sscp_memory_scope::device:
      return __HIP_MEMORY_SCOPE_AGENT;
    case __hipsycl_sscp_memory_scope::system:
      return __HIP_MEMORY_SCOPE_SYSTEM;
  }
}



// ********************** atomic store ***************************

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN void __hipsycl_sscp_atomic_store_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __atomic_store_n(ptr, x, builtin_memory_order(order));
}


// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_load_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr) {
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_load_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr) {
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_load_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr) {
  return __atomic_load_n(ptr, builtin_memory_order(order));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_load_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr) {
  return __atomic_load_n(ptr, builtin_memory_order(order));
}


// ********************** atomic exchange ***************************

HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_exchange_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_exchange(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_exchange_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr,
    __hipsycl_int16 x) {
  return __hip_atomic_exchange(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_exchange_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr,
    __hipsycl_int32 x) {
  return __hip_atomic_exchange(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_exchange_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr,
    __hipsycl_int64 x) {
  return __hip_atomic_exchange(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
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

  return __hip_atomic_compare_exchange_strong(
      ptr, expected, desired, builtin_memory_order(success),
      builtin_memory_order(failure), builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int16 *ptr, __hipsycl_int16 *expected, __hipsycl_int16 desired) {

  return __hip_atomic_compare_exchange_strong(
      ptr, expected, desired, builtin_memory_order(success),
      builtin_memory_order(failure), builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int32 *ptr, __hipsycl_int32 *expected, __hipsycl_int32 desired) {

  return __hip_atomic_compare_exchange_strong(
      ptr, expected, desired, builtin_memory_order(success),
      builtin_memory_order(failure), builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN bool __hipsycl_sscp_cmp_exch_strong_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order success,
    __hipsycl_sscp_memory_order failure, __hipsycl_sscp_memory_scope scope,
    __hipsycl_int64 *ptr, __hipsycl_int64 *expected, __hipsycl_int64 desired) {

  return __hip_atomic_compare_exchange_strong(
      ptr, expected, desired, builtin_memory_order(success),
      builtin_memory_order(failure), builtin_memory_scope(scope));
}

// "The boughs so old
//  With leaves of gold
//  In autumn's cloak
//  The endless oak."
//
//  Today's song recommendation: Empyrium - The Oaken Throne.


HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_and_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {

  return __hip_atomic_fetch_and(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_and_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {

  return __hip_atomic_fetch_and(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_and_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {

  return __hip_atomic_fetch_and(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_and_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {

  return __hip_atomic_fetch_and(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_or_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_or(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_or_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_or(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_or_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_or(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_or_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_or(ptr, x, builtin_memory_order(order),
                               builtin_memory_scope(scope));
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_xor_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_xor(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_xor_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_xor(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_xor_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_xor(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_xor_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_xor(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}



// TODO: It seems that at least on gfx90a, there are additional unsafe atomic
// operations that can yield a performance improvement. Should we expose those?
HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_add_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_add_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_add_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_add_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_add_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_add_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_add_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_add_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_add_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_add_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_sub_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_sub_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_sub_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_sub_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_sub_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_sub_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_sub_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_sub_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_sub_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_sub_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  return __hip_atomic_fetch_add(ptr, -x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_min_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_min_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_min_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_min_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_min_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_min_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_min_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_min_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_min_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_min_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  return __hip_atomic_fetch_min(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}



HIPSYCL_SSCP_BUILTIN __hipsycl_int8 __hipsycl_sscp_atomic_fetch_max_i8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int8 *ptr, __hipsycl_int8 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int16 __hipsycl_sscp_atomic_fetch_max_i16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int16 *ptr, __hipsycl_int16 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_atomic_fetch_max_i32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int32 *ptr, __hipsycl_int32 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_int64 __hipsycl_sscp_atomic_fetch_max_i64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_int64 *ptr, __hipsycl_int64 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_atomic_fetch_max_u8(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint8 *ptr, __hipsycl_uint8 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_atomic_fetch_max_u16(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint16 *ptr, __hipsycl_uint16 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_atomic_fetch_max_u32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint32 *ptr, __hipsycl_uint32 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_atomic_fetch_max_u64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_uint64 *ptr, __hipsycl_uint64 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f32 __hipsycl_sscp_atomic_fetch_max_f32(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f32 *ptr, __hipsycl_f32 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

HIPSYCL_SSCP_BUILTIN __hipsycl_f64 __hipsycl_sscp_atomic_fetch_max_f64(
    __hipsycl_sscp_address_space as, __hipsycl_sscp_memory_order order,
    __hipsycl_sscp_memory_scope scope, __hipsycl_f64 *ptr, __hipsycl_f64 x) {
  return __hip_atomic_fetch_max(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}


