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

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_BUILTIN i8  __acpp_sscp_metal_atomic_load_i8(const char* s, i8* ptr);
HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_metal_atomic_load_i16(const char* s, i16* ptr);
HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_metal_atomic_load_i32(const char* s, i32* ptr);
HIPSYCL_SSCP_BUILTIN u8  __acpp_sscp_metal_atomic_load_u8(const char* s, u8* ptr);
HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_metal_atomic_load_u16(const char* s, u16* ptr);
HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_metal_atomic_load_u32(const char* s, u32* ptr);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_atomic_load_f32(const char* s, f32* ptr);

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_i8(const char* s, i8* ptr, i8 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_i16(const char* s, i16* ptr, i16 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_i32(const char* s, i32* ptr, i32 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_u8(const char* s, u8* ptr, u8 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_u16(const char* s, u16* ptr, u16 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_u32(const char* s, u32* ptr, u32 val);
HIPSYCL_SSCP_BUILTIN void __acpp_sscp_metal_atomic_store_f32(const char* s, f32* ptr, f32 val);

HIPSYCL_SSCP_BUILTIN i8  __acpp_sscp_metal_atomic_exchange_i8(const char* s, i8* ptr, i8 val);
HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_metal_atomic_exchange_i16(const char* s, i16* ptr, i16 val);
HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_metal_atomic_exchange_i32(const char* s, i32* ptr, i32 val);
HIPSYCL_SSCP_BUILTIN u8  __acpp_sscp_metal_atomic_exchange_u8(const char* s, u8* ptr, u8 val);
HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_metal_atomic_exchange_u16(const char* s, u16* ptr, u16 val);
HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_metal_atomic_exchange_u32(const char* s, u32* ptr, u32 val);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_atomic_exchange_f32(const char* s, f32* ptr, f32 val);

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_i8(const char* s, i8* ptr, i8* expected, i8 desired);
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_i16(const char* s, i16* ptr, i16* expected, i16 desired);
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_i32(const char* s, i32* ptr, i32* expected, i32 desired);
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_u8(const char* s, u8* ptr, u8* expected, u8 desired);
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_u16(const char* s, u16* ptr, u16* expected, u16 desired);
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_metal_atomic_cmpxchg_u32(const char* s, u32* ptr, u32* expected, u32 desired);

HIPSYCL_SSCP_BUILTIN i8  __acpp_sscp_metal_atomic_fetch_i8(const char* s, i8* ptr, i8 val);
HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_metal_atomic_fetch_i16(const char* s, i16* ptr, i16 val);
HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_metal_atomic_fetch_i32(const char* s, i32* ptr, i32 val);
HIPSYCL_SSCP_BUILTIN u8  __acpp_sscp_metal_atomic_fetch_u8(const char* s, u8* ptr, u8 val);
HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_metal_atomic_fetch_u16(const char* s, u16* ptr, u16 val);
HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_metal_atomic_fetch_u32(const char* s, u32* ptr, u32 val);
HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_metal_atomic_fetch_f32(const char* s, f32* ptr, f32 val);

// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_load_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr) {
  return __acpp_sscp_metal_atomic_load_i8("atomic_load_explicit(__atomic_pointer_cast<char>(%s), memory_order_relaxed)", ptr);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_load_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr) {
  return __acpp_sscp_metal_atomic_load_i16("atomic_load_explicit(__atomic_pointer_cast<short>(%s), memory_order_relaxed)", ptr);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr) {
  return __acpp_sscp_metal_atomic_load_i32("atomic_load_explicit(__atomic_pointer_cast<int>(%s), memory_order_relaxed)", ptr);
}

// ********************** atomic store ***************************

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  __acpp_sscp_metal_atomic_store_i8("atomic_store_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  __acpp_sscp_metal_atomic_store_i16("atomic_store_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  __acpp_sscp_metal_atomic_store_i32("atomic_store_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************** atomic exchange ***************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_exchange_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_exchange_i8("atomic_exchange_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_exchange_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_exchange_i16("atomic_exchange_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_exchange_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_exchange_i32("atomic_exchange_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************** atomic compare exchange weak **********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i8 *ptr, i8 *expected, i8 desired) {
  return __acpp_sscp_metal_atomic_cmpxchg_i8(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<char>(%s), __pointer_cast<char>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i8 *ptr, i8 *expected, i8 desired) {
  i8 old = *expected;
  while (!__acpp_sscp_metal_atomic_cmpxchg_i8(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<char>(%s), __pointer_cast<char>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired)) {
    if (*expected != old) return false;
  }
  return true;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i16 *ptr, i16 *expected, i16 desired) {
  return __acpp_sscp_metal_atomic_cmpxchg_i16(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<short>(%s), __pointer_cast<short>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired);
}

// ********************* atomic compare exchange strong  *********************

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i16 *ptr, i16 *expected, i16 desired) {
  i16 old = *expected;
  while (!__acpp_sscp_metal_atomic_cmpxchg_i16(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<short>(%s), __pointer_cast<short>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired)) {
    if (*expected != old) return false;
  }
  return true;
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i32 *ptr, i32 *expected, i32 desired)
{
  return __acpp_sscp_metal_atomic_cmpxchg_i32(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<int>(%s), __pointer_cast<int>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired);
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    i32 *ptr, i32 *expected, i32 desired)
{
  i32 old = *expected;
  while (!__acpp_sscp_metal_atomic_cmpxchg_i32(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<int>(%s), __pointer_cast<int>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    ptr, expected, desired)) {
    if (*expected != old) return false;
  }
  return true;
}

// ********************* atomic fetch add ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_add_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_add_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_add_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_add_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_add_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_add_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u8 __acpp_sscp_atomic_fetch_add_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u8 *ptr, u8 x) {
  return __acpp_sscp_metal_atomic_fetch_u8("atomic_fetch_add_explicit(__atomic_pointer_cast<uchar>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_atomic_fetch_add_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u16 *ptr, u16 x) {
  return __acpp_sscp_metal_atomic_fetch_u16("atomic_fetch_add_explicit(__atomic_pointer_cast<ushort>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_atomic_fetch_add_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u32 *ptr, u32 x) {
  return __acpp_sscp_metal_atomic_fetch_u32("atomic_fetch_add_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, f32 *ptr, f32 x) {
  return __acpp_sscp_metal_atomic_fetch_f32("atomic_fetch_add_explicit(__atomic_pointer_cast<float>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************* atomic fetch sub ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_sub_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_sub_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_sub_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_sub_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_sub_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_sub_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u8 __acpp_sscp_atomic_fetch_sub_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u8 *ptr, u8 x) {
  return __acpp_sscp_metal_atomic_fetch_u8("atomic_fetch_sub_explicit(__atomic_pointer_cast<uchar>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_atomic_fetch_sub_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u16 *ptr, u16 x) {
  return __acpp_sscp_metal_atomic_fetch_u16("atomic_fetch_sub_explicit(__atomic_pointer_cast<ushort>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_atomic_fetch_sub_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u32 *ptr, u32 x) {
  return __acpp_sscp_metal_atomic_fetch_u32("atomic_fetch_sub_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atomic_fetch_sub_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, f32 *ptr, f32 x) {
  return __acpp_sscp_metal_atomic_fetch_f32("atomic_fetch_sub_explicit(__atomic_pointer_cast<float>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************* atomic fetch and ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_and_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_and_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_and_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_and_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_and_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_and_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************* atomic fetch or *************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_or_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_or_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_or_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_or_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_or_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_or_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************* atomic fetch xor ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_xor_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_xor_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_xor_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_xor_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_xor_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_xor_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

// ********************* atomic fetch min ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_min_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_min_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_min_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_min_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_min_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_min_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u8 __acpp_sscp_atomic_fetch_min_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u8 *ptr, u8 x) {
  return __acpp_sscp_metal_atomic_fetch_u8("atomic_fetch_min_explicit(__atomic_pointer_cast<uchar>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_atomic_fetch_min_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u16 *ptr, u16 x) {
  return __acpp_sscp_metal_atomic_fetch_u16("atomic_fetch_min_explicit(__atomic_pointer_cast<ushort>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_atomic_fetch_min_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u32 *ptr, u32 x) {
  return __acpp_sscp_metal_atomic_fetch_u32("atomic_fetch_min_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atomic_fetch_min_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, f32 *ptr, f32 operand) {
  u32* addr = (u32*)ptr;
  u32 old_bits = __acpp_sscp_metal_atomic_load_u32("atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", addr);

  while (true) {
    f32 old_val = *((f32*)&old_bits);
    if (old_val <= operand) {
      return old_val;
    }
    f32 new_val = operand;
    u32 new_bits = *((u32*)&new_val);
    u32 expected = old_bits;
    bool ok = __acpp_sscp_metal_atomic_cmpxchg_u32(
      "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<uint>(%s), __pointer_cast<uint>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
      addr, &expected, new_bits);
    if (ok) {
      return old_val;
    }
    old_bits = expected;
  }
}

// ********************* atomic fetch max ************************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_fetch_max_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  return __acpp_sscp_metal_atomic_fetch_i8("atomic_fetch_max_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_fetch_max_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  return __acpp_sscp_metal_atomic_fetch_i16("atomic_fetch_max_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_fetch_max_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  return __acpp_sscp_metal_atomic_fetch_i32("atomic_fetch_max_explicit(__atomic_pointer_cast<int>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u8 __acpp_sscp_atomic_fetch_max_u8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u8 *ptr, u8 x) {
  return __acpp_sscp_metal_atomic_fetch_u8("atomic_fetch_max_explicit(__atomic_pointer_cast<uchar>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u16 __acpp_sscp_atomic_fetch_max_u16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u16 *ptr, u16 x) {
  return __acpp_sscp_metal_atomic_fetch_u16("atomic_fetch_max_explicit(__atomic_pointer_cast<ushort>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_atomic_fetch_max_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, u32 *ptr, u32 x) {
  return __acpp_sscp_metal_atomic_fetch_u32("atomic_fetch_max_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN f32 __acpp_sscp_atomic_fetch_max_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, f32 *ptr, f32 operand) {
  u32* addr = (u32*)ptr;
  u32 old_bits = __acpp_sscp_metal_atomic_load_u32("atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", addr);

  while (true) {
    f32 old_val = *((f32*)&old_bits);
    if (old_val >= operand) {
      return old_val;
    }
    f32 new_val = operand;
    u32 new_bits = *((u32*)&new_val);
    u32 expected = old_bits;
    bool ok = __acpp_sscp_metal_atomic_cmpxchg_u32(
      "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<uint>(%s), __pointer_cast<uint>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
      addr, &expected, new_bits);
    if (ok) {
      return old_val;
    }
    old_bits = expected;
  }
}
