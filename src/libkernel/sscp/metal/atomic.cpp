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
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
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

namespace {

inline void fence_before_atomic(__acpp_sscp_memory_order order, __acpp_sscp_memory_scope scope) {
  if (order == __acpp_sscp_memory_order::release ||
      order == __acpp_sscp_memory_order::acq_rel ||
      order == __acpp_sscp_memory_order::seq_cst) {
    __acpp_sscp_memory_fence(scope, order);
  }
}

inline void fence_after_atomic(__acpp_sscp_memory_order order, __acpp_sscp_memory_scope scope) {
  if (order == __acpp_sscp_memory_order::acquire ||
      order == __acpp_sscp_memory_order::acq_rel ||
      order == __acpp_sscp_memory_order::seq_cst) {
    __acpp_sscp_memory_fence(scope, order);
  }
}

} // namespace

// ********************** atomic load ***************************

HIPSYCL_SSCP_BUILTIN i8 __acpp_sscp_atomic_load_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr) {
  i8 result = __acpp_sscp_metal_atomic_load_i8("atomic_load_explicit(__atomic_pointer_cast<char>(%s), memory_order_relaxed)", ptr);
  fence_after_atomic(order, scope);
  return result;
}

HIPSYCL_SSCP_BUILTIN i16 __acpp_sscp_atomic_load_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr) {
  i16 result = __acpp_sscp_metal_atomic_load_i16("atomic_load_explicit(__atomic_pointer_cast<short>(%s), memory_order_relaxed)", ptr);
  fence_after_atomic(order, scope);
  return result;
}

HIPSYCL_SSCP_BUILTIN i32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr) {
  i32 result = __acpp_sscp_metal_atomic_load_i32("atomic_load_explicit(__atomic_pointer_cast<int>(%s), memory_order_relaxed)", ptr);
  fence_after_atomic(order, scope);
  return result;
}

// ********************** atomic store ***************************

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i8(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i8 *ptr, i8 x) {
  fence_before_atomic(order, scope);
  __acpp_sscp_metal_atomic_store_i8("atomic_store_explicit(__atomic_pointer_cast<char>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i16(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i16 *ptr, i16 x) {
  fence_before_atomic(order, scope);
  __acpp_sscp_metal_atomic_store_i16("atomic_store_explicit(__atomic_pointer_cast<short>(%s), %s, memory_order_relaxed)", ptr, x);
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, i32 *ptr, i32 x) {
  fence_before_atomic(order, scope);
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

  if (as == __acpp_sscp_address_space::local_space) {
    u32 *addr = (u32 *)ptr;
    u32 old_bits = __acpp_sscp_metal_atomic_load_u32(
      "atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", addr);
    while (true) {
      f32 old_val = *((f32 *)&old_bits);
      f32 new_val = old_val + x;
      u32 new_bits = *((u32 *)&new_val);
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
  return __acpp_sscp_metal_atomic_fetch_f32(
    "atomic_fetch_add_explicit(__atomic_pointer_cast<float>(%s), %s, memory_order_relaxed)", ptr, x);
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

  if (as == __acpp_sscp_address_space::local_space) {
    u32 *addr = (u32 *)ptr;
    u32 old_bits = __acpp_sscp_metal_atomic_load_u32(
      "atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", addr);

    while (true) {
      f32 old_val = *((f32 *)&old_bits);
      f32 new_val = old_val - x;
      u32 new_bits = *((u32 *)&new_val);
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
  return __acpp_sscp_metal_atomic_fetch_f32(
    "atomic_fetch_sub_explicit(__atomic_pointer_cast<float>(%s), %s, memory_order_relaxed)", ptr, x);
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

// Metal lacks general 64-bit atomics. Use hashed 32-bit locks in device memory,
// including for objects in threadgroup memory
// The runtime binds a table shared across kernels at buffer(2)
HIPSYCL_SSCP_BUILTIN u32* __acpp_sscp_metal_symbol_atomic64_lock_base(const char* s);

namespace {

// Must match metal_allocator::atomic64_lock_table_size
constexpr u64 atomic64_lock_table_size = 16384;

inline u32* atomic64_lock_for(const void* ptr) {
  u32* base = __acpp_sscp_metal_symbol_atomic64_lock_base("__acpp_sscp_metal_atomic64_lock_base");
  u64 address = (u64)ptr;
  u64 hash = ((address >> 3) ^ (address >> 17)) & (atomic64_lock_table_size - 1);
  return base + hash;
}

inline bool atomic64_try_lock(u32* lock) {
  u32 expected = 0;
  return __acpp_sscp_metal_atomic_cmpxchg_u32(
    "atomic_compare_exchange_weak_explicit(__atomic_pointer_cast<uint>(%s), __pointer_cast<uint>(%s), %s, memory_order_relaxed, memory_order_relaxed)",
    lock, &expected, (u32)1);
}

inline void atomic64_unlock(u32* lock) {
  __acpp_sscp_metal_atomic_store_u32(
    "atomic_store_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)",
    lock, (u32)0);
}

inline void atomic64_read_halves(u64* ptr, u32& low, u32& high) {
  u32* lower = (u32*)ptr;
  low = __acpp_sscp_metal_atomic_load_u32("atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", lower);
  high = __acpp_sscp_metal_atomic_load_u32("atomic_load_explicit(__atomic_pointer_cast<uint>(%s), memory_order_relaxed)", lower + 1);
}

inline void atomic64_write_halves(u64* ptr, u32 low, u32 high) {
  u32* lower = (u32*)ptr;
  __acpp_sscp_metal_atomic_store_u32("atomic_store_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", lower, low);
  __acpp_sscp_metal_atomic_store_u32("atomic_store_explicit(__atomic_pointer_cast<uint>(%s), %s, memory_order_relaxed)", lower + 1, high);
}

// Every operation of the ordered pass works on 32-bit halves and never forms a
// 64-bit value inside the loop: some Metal devices cannot compile that, and
// whether the optimizer splits a 64-bit operation by itself depends on the
// surrounding control flow.
template<class F>
inline u64 atomic64_update_halves(
  u64* ptr, u32 x_low, u32 x_high,
  __acpp_sscp_memory_scope scope, F f)
{
  const u32 my_lane = __acpp_sscp_get_subgroup_local_id();
  const u32 lanes = __acpp_sscp_get_subgroup_max_size();
  u32 old_low = 0;
  u32 old_high = 0;
  for (u32 lane = 0; lane < lanes; ++lane) {
    if (lane == my_lane) {
      u32* lock = atomic64_lock_for(ptr);
      while (!atomic64_try_lock(lock)) {
        ;
      }
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_read_halves(ptr, old_low, old_high);
      u32 new_low = 0;
      u32 new_high = 0;
      f(old_low, old_high, x_low, x_high, new_low, new_high);
      atomic64_write_halves(ptr, new_low, new_high);
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_unlock(lock);
    }
  }
  return ((u64)old_high << 32) | (u64)old_low;
}

// min/max/cmpexchg use 32-bit halves because some metal-devices
// cannot compile 64-bit comparisons in the loop (e.g. virtual GPUs)
inline bool atomic64_less(u32 a_low, u32 a_high, u32 b_low, u32 b_high, bool is_signed) {
  if (a_high != b_high) {
    return is_signed ? (i32)a_high < (i32)b_high : a_high < b_high;
  }
  return a_low < b_low;
}

inline u64 atomic64_min_max(
  u64* ptr, u64 x, __acpp_sscp_memory_scope scope,
  bool take_max, bool is_signed)
{
  const u32 x_low = (u32)x;
  const u32 x_high = (u32)(x >> 32);
  const u32 my_lane = __acpp_sscp_get_subgroup_local_id();
  const u32 lanes = __acpp_sscp_get_subgroup_max_size();
  u32 old_low = 0;
  u32 old_high = 0;
  for (u32 lane = 0; lane < lanes; ++lane) {
    if (lane == my_lane) {
      u32* lock = atomic64_lock_for(ptr);
      while (!atomic64_try_lock(lock)) {
        ;
      }
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_read_halves(ptr, old_low, old_high);
      const bool replace = take_max
        ? atomic64_less(old_low, old_high, x_low, x_high, is_signed)
        : atomic64_less(x_low, x_high, old_low, old_high, is_signed);
      if (replace) {
        atomic64_write_halves(ptr, x_low, x_high);
      }
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_unlock(lock);
    }
  }
  return ((u64)old_high << 32) | (u64)old_low;
}

inline bool atomic64_compare_exchange(
  u64* ptr, u64* expected, u64 desired,
  __acpp_sscp_memory_scope scope)
{
  const u32 desired_low = (u32)desired;
  const u32 desired_high = (u32)(desired >> 32);
  const u32 expected_low = (u32)*expected;
  const u32 expected_high = (u32)(*expected >> 32);
  const u32 my_lane = __acpp_sscp_get_subgroup_local_id();
  const u32 lanes = __acpp_sscp_get_subgroup_max_size();
  bool success = false;
  u32 old_low = 0;
  u32 old_high = 0;
  for (u32 lane = 0; lane < lanes; ++lane) {
    if (lane == my_lane) {
      u32* lock = atomic64_lock_for(ptr);
      while (!atomic64_try_lock(lock)) {
        ;
      }
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_read_halves(ptr, old_low, old_high);
      success = old_low == expected_low && old_high == expected_high;
      if (success) {
        atomic64_write_halves(ptr, desired_low, desired_high);
      }
      __acpp_sscp_memory_fence(scope, __acpp_sscp_memory_order::seq_cst);
      atomic64_unlock(lock);
    }
  }
  if (!success) {
    *expected = ((u64)old_high << 32) | (u64)old_low;
  }
  return success;
}

} // namespace

#define ACPP_ATOMIC64_HALVES(op, type, low_expr, high_expr) \
HIPSYCL_SSCP_BUILTIN type __acpp_sscp_atomic_##op##_##type( \
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order, \
  __acpp_sscp_memory_scope scope, type *ptr, type x) { \
  return (type)atomic64_update_halves((u64*)ptr, (u32)(u64)x, (u32)((u64)x >> 32), scope, \
    [](u32 old_low, u32 old_high, u32 x_low, u32 x_high, u32& new_low, u32& new_high) { \
      new_low = (low_expr); \
      new_high = (high_expr); \
    }); \
}

ACPP_ATOMIC64_HALVES(exchange,  i64, x_low, x_high)
ACPP_ATOMIC64_HALVES(fetch_and, i64, old_low & x_low, old_high & x_high)
ACPP_ATOMIC64_HALVES(fetch_or,  i64, old_low | x_low, old_high | x_high)
ACPP_ATOMIC64_HALVES(fetch_xor, i64, old_low ^ x_low, old_high ^ x_high)

#undef ACPP_ATOMIC64_HALVES

#define ACPP_ATOMIC64_MIN_MAX(op, type, take_max, is_signed) \
HIPSYCL_SSCP_BUILTIN type __acpp_sscp_atomic_##op##_##type( \
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order, \
  __acpp_sscp_memory_scope scope, type *ptr, type x) { \
  return (type)atomic64_min_max((u64*)ptr, (u64)x, scope, take_max, is_signed); \
}

ACPP_ATOMIC64_MIN_MAX(fetch_min, i64, false, true)
ACPP_ATOMIC64_MIN_MAX(fetch_min, u64, false, false)
ACPP_ATOMIC64_MIN_MAX(fetch_max, i64, true, true)
ACPP_ATOMIC64_MIN_MAX(fetch_max, u64, true, false)

#undef ACPP_ATOMIC64_MIN_MAX

// signed/unsigned addition/subtraction share the same bitwise operation
// add the halves with an explicit carry
#define ACPP_ATOMIC64_ADD(op, type, operand) \
HIPSYCL_SSCP_BUILTIN type __acpp_sscp_atomic_##op##_##type( \
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order, \
  __acpp_sscp_memory_scope scope, type *ptr, type x) { \
  const u64 value = (operand); \
  return (type)atomic64_update_halves((u64*)ptr, (u32)value, (u32)(value >> 32), scope, \
    [](u32 old_low, u32 old_high, u32 x_low, u32 x_high, u32& new_low, u32& new_high) { \
      new_low = old_low + x_low; \
      new_high = old_high + x_high + (new_low < old_low ? 1u : 0u); \
    }); \
}

ACPP_ATOMIC64_ADD(fetch_add, i64, (u64)x)
ACPP_ATOMIC64_ADD(fetch_add, u64, x)
ACPP_ATOMIC64_ADD(fetch_sub, i64, (u64)(-(i64)x))
ACPP_ATOMIC64_ADD(fetch_sub, u64, (u64)(0 - x))

#undef ACPP_ATOMIC64_ADD

#define ACPP_ATOMIC64_COMPARE_EXCHANGE(op) \
HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_##op##_i64( \
  __acpp_sscp_address_space as, __acpp_sscp_memory_order success, \
  __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope, \
  i64 *ptr, i64 *expected, i64 desired) { \
  return atomic64_compare_exchange((u64*)ptr, (u64*)expected, (u64)desired, scope); \
}

// The lock makes both forms equally strong, so weak cannot fail spuriously
ACPP_ATOMIC64_COMPARE_EXCHANGE(weak)
ACPP_ATOMIC64_COMPARE_EXCHANGE(strong)

#undef ACPP_ATOMIC64_COMPARE_EXCHANGE

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i64(
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
  __acpp_sscp_memory_scope scope, i64 *ptr, i64 x) {
  atomic64_update_halves((u64*)ptr, (u32)(u64)x, (u32)((u64)x >> 32), scope,
    [](u32, u32, u32 x_low, u32 x_high, u32& new_low, u32& new_high) {
      new_low = x_low;
      new_high = x_high;
    });
}

HIPSYCL_SSCP_BUILTIN i64 __acpp_sscp_atomic_load_i64(
  __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
  __acpp_sscp_memory_scope scope, i64 *ptr) {
  return (i64)atomic64_update_halves((u64*)ptr, 0, 0, scope,
    [](u32 old_low, u32 old_high, u32, u32, u32& new_low, u32& new_high) {
      new_low = old_low;
      new_high = old_high;
    });
}
