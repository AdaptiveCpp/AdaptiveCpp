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

// Taken from the mapping from OpenCL-C to LLVM-IR from clang
inline constexpr int builtin_memory_order(__acpp_sscp_memory_order o) noexcept {
  switch (o) {
  case __acpp_sscp_memory_order::relaxed:
    return 0;
  case __acpp_sscp_memory_order::acquire:
    return 2;
  case __acpp_sscp_memory_order::release:
    return 3;
  case __acpp_sscp_memory_order::acq_rel:
    return 4;
  case __acpp_sscp_memory_order::seq_cst:
    return 5;
  }
  return 0;
}

// Taken from the mapping from OpenCL-C to LLVM-IR from clang
inline constexpr int builtin_memory_scope(__acpp_sscp_memory_scope s) noexcept {
  switch (s) {
  case __acpp_sscp_memory_scope::work_item:
    return 0;
  case __acpp_sscp_memory_scope::work_group:
    return 1;
  case __acpp_sscp_memory_scope::sub_group:
    return 4;
  case __acpp_sscp_memory_scope::device:
    return 2;
  case __acpp_sscp_memory_scope::system:
    return 3; // memory_scope_all_devices
  }
  return 0;
}

// ********************** atomic store ***************************

extern "C" {
// Address space(1), int32
void _Z21atomic_store_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(1), int32
void _Z21atomic_store_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
void _Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
void _Z21atomic_store_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(1), int64
void _Z21atomic_store_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
void _Z21atomic_store_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z21atomic_store_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z21atomic_store_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z21atomic_store_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN void __acpp_sscp_atomic_store_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z21atomic_store_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z21atomic_store_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z21atomic_store_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

// ********************** atomic load ***************************

extern "C" {
// Address space(1), int32
__acpp_int32
_Z20atomic_load_explicitPU3AS1VU7_Atomici12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z20atomic_load_explicitPU3AS2VU7_Atomici12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z20atomic_load_explicitPU3AS1VU7_Atomicl12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z20atomic_load_explicitPU3AS2VU7_Atomicl12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z20atomic_load_explicitPU3AS4VU7_Atomicl12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_load_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z20atomic_load_explicitPU3AS1VU7_Atomici12memory_order12memory_scope(
        ptr, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z20atomic_load_explicitPU3AS2VU7_Atomici12memory_order12memory_scope(
        ptr, o, s);
  } else {
    return _Z20atomic_load_explicitPU3AS4VU7_Atomici12memory_order12memory_scope(
        ptr, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_load_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z20atomic_load_explicitPU3AS1VU7_Atomicl12memory_order12memory_scope(
        ptr, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z20atomic_load_explicitPU3AS2VU7_Atomicl12memory_order12memory_scope(
        ptr, o, s);
  } else {
    return _Z20atomic_load_explicitPU3AS4VU7_Atomicl12memory_order12memory_scope(
        ptr, o, s);
  }
}

// ********************** atomic exchange ***************************

extern "C" {
// Address space(1), int32
__acpp_int32
_Z24atomic_exchange_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __attribute__((address_space(1))) __acpp_int32 *ptr, __acpp_int32 x,
    __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z24atomic_exchange_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __attribute__((address_space(2))) __acpp_int32 *ptr, __acpp_int32 x,
    __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z24atomic_exchange_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z24atomic_exchange_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __attribute__((address_space(1))) __acpp_int64 *ptr, __acpp_int64 x,
    __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z24atomic_exchange_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __attribute__((address_space(2))) __acpp_int64 *ptr, __acpp_int64 x,
    __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z24atomic_exchange_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_exchange_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z24atomic_exchange_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        (__attribute__((address_space(1))) __acpp_int32 *)ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z24atomic_exchange_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        (__attribute__((address_space(2))) __acpp_int32 *)ptr, x, o, s);
  } else {
    return _Z24atomic_exchange_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_exchange_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z24atomic_exchange_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        (__attribute__((address_space(1))) __acpp_int64 *)ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z24atomic_exchange_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        (__attribute__((address_space(2))) __acpp_int64 *)ptr, x, o, s);
  } else {
    return _Z24atomic_exchange_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

// ********************** atomic compare exchange weak **********************
extern "C" {
// Address space(1), int32
bool _Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(2), int32
bool _Z37atomic_compare_exchange_weak_explicitPU3AS2VU7_AtomiciPU3AS2ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(4), int32
bool _Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);

// Address space(1), int64
bool _Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(2), int64
bool _Z37atomic_compare_exchange_weak_explicitPU3AS2VU7_AtomiciPU3AS2ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(4), int64
bool _Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {
  auto so = builtin_memory_order(success);
  auto fo = builtin_memory_order(failure);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS2VU7_AtomiciPU3AS2ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  }
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_weak_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {
  auto so = builtin_memory_order(success);
  auto fo = builtin_memory_order(failure);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS1VU7_AtomiciPU3AS1ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS2VU7_AtomiciPU3AS2ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else {
    return _Z37atomic_compare_exchange_weak_explicitPU3AS4VU7_AtomiciPU3AS4ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  }
}

// ********************* atomic compare exchange strong  *********************
extern "C" {
// Address space(1), int32
bool _Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(2), int32
bool _Z39atomic_compare_exchange_strong_explicitPU3AS2VU7_AtomiciPU3AS2ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(4), int32
bool _Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);

// Address space(1), int64
bool _Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(2), int64
bool _Z39atomic_compare_exchange_strong_explicitPU3AS2VU7_AtomiciPU3AS2ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
// Address space(4), int64
bool _Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ll12memory_orderS4_12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired,
    __acpp_int32 success, __acpp_int32 failure, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int32 *ptr, __acpp_int32 *expected, __acpp_int32 desired) {
  auto so = builtin_memory_order(success);
  auto fo = builtin_memory_order(failure);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS2VU7_AtomiciPU3AS2ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ii12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  }
}

HIPSYCL_SSCP_BUILTIN bool __acpp_sscp_cmp_exch_strong_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order success,
    __acpp_sscp_memory_order failure, __acpp_sscp_memory_scope scope,
    __acpp_int64 *ptr, __acpp_int64 *expected, __acpp_int64 desired) {
  auto so = builtin_memory_order(success);
  auto fo = builtin_memory_order(failure);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS1VU7_AtomiciPU3AS1ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS2VU7_AtomiciPU3AS2ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  } else {
    return _Z39atomic_compare_exchange_strong_explicitPU3AS4VU7_AtomiciPU3AS4ll12memory_orderS4_12memory_scope(
        ptr, expected, desired, so, fo, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_and_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_and_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_and_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_and_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_and_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_and_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_and_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_and_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z24atomic_fetch_or_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z24atomic_fetch_or_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_or_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z24atomic_fetch_or_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_or_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z24atomic_fetch_or_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z24atomic_fetch_or_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z24atomic_fetch_or_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_xor_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_xor_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_xor_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_xor_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_xor_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_xor_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_xor_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_xor_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint32
__acpp_uint32
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), uint32
__acpp_uint32
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint32
__acpp_uint32
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint64
__acpp_uint64
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), int64
__acpp_uint64
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint64
__acpp_uint64
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), f32
__acpp_f32
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), f32
__acpp_f32
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f32
__acpp_f32
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), f64
__acpp_f64
_Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope64);
// Address space(2), f64
__acpp_f64
_Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f64
__acpp_f64
_Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_add_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_add_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_add_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_add_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_add_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_add_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_add_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint32
__acpp_uint32
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), uint32
__acpp_uint32
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint32
__acpp_uint32
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint64
__acpp_uint64
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), int64
__acpp_uint64
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint64
__acpp_uint64
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), f32
__acpp_f32
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), f32
__acpp_f32
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f32
__acpp_f32
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), f64
__acpp_f64
_Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope64);
// Address space(2), f64
__acpp_f64
_Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f64
__acpp_f64
_Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_sub_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_sub_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_sub_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_sub_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_sub_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_sub_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_sub_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_sub_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint32
__acpp_uint32
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), uint32
__acpp_uint32
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint32
__acpp_uint32
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint64
__acpp_uint64
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), int64
__acpp_uint64
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint64
__acpp_uint64
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), f32
__acpp_f32
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), f32
__acpp_f32
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f32
__acpp_f32
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), f64
__acpp_f64
_Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope64);
// Address space(2), f64
__acpp_f64
_Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f64
__acpp_f64
_Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_min_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_min_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_min_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_min_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_min_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_min_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_min_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_min_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_min_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

extern "C" {
// Address space(1), int32
__acpp_int32
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int32
__acpp_int32
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int32
__acpp_int32
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
    __acpp_int32 *ptr, __acpp_int32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint32
__acpp_uint32
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), uint32
__acpp_uint32
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint32
__acpp_uint32
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
    __acpp_uint32 *ptr, __acpp_uint32 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), int64
__acpp_int64
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), int64
__acpp_int64
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), int64
__acpp_int64
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
    __acpp_int64 *ptr, __acpp_int64 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), uint64
__acpp_uint64
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(2), int64
__acpp_uint64
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);
// Address space(4), uint64
__acpp_uint64
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
    __acpp_uint64 *ptr, __acpp_uint64 x, __acpp_int32 order,
    __acpp_int32 scope);

// Address space(1), f32
__acpp_f32
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(2), f32
__acpp_f32
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f32
__acpp_f32
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
    __acpp_f32 *ptr, __acpp_f32 x, __acpp_int32 order, __acpp_int32 scope);

// Address space(1), f64
__acpp_f64
_Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope64);
// Address space(2), f64
__acpp_f64
_Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
// Address space(4), f64
__acpp_f64
_Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
    __acpp_f64 *ptr, __acpp_f64 x, __acpp_int32 order, __acpp_int32 scope);
};

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_atomic_fetch_max_i32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int32 *ptr, __acpp_int32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicii12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_int64 __acpp_sscp_atomic_fetch_max_i64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_int64 *ptr, __acpp_int64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicll12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_atomic_fetch_max_u32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint32 *ptr, __acpp_uint32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicjj12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_atomic_fetch_max_u64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_uint64 *ptr, __acpp_uint64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicmm12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_atomic_fetch_max_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicff12memory_order12memory_scope(
        ptr, x, o, s);
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_atomic_fetch_max_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
  auto o = builtin_memory_order(order);
  auto s = builtin_memory_scope(scope);
  if (as == __acpp_sscp_address_space::global_space) {
    return _Z25atomic_fetch_max_explicitPU3AS1VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else if (as == __acpp_sscp_address_space::local_space) {
    return _Z25atomic_fetch_max_explicitPU3AS2VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  } else {
    return _Z25atomic_fetch_max_explicitPU3AS4VU7_Atomicdd12memory_order12memory_scope(
        ptr, x, o, s);
  }
}
