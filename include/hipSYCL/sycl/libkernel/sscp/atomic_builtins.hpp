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
#ifndef HIPSYCL_ATOMIC_SSCP_BUILTINS_INTERFACE_HPP
#define HIPSYCL_ATOMIC_SSCP_BUILTINS_INTERFACE_HPP

#include <type_traits>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/detail/util.hpp"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP

#include "builtins/atomic.hpp"

namespace hipsycl {
namespace sycl {
namespace detail::sscp_builtins {

#define return_cast(x) return __builtin_bit_cast(T, x);

template <access::address_space S, class T>
HIPSYCL_BUILTIN void __acpp_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    __acpp_sscp_atomic_store_i8(S, order, scope,
                                   reinterpret_cast<__acpp_int8 *>(addr),
                                   __builtin_bit_cast(__acpp_int8, x));
  } else if constexpr (sizeof(T) == 2) {
    __acpp_sscp_atomic_store_i16(S, order, scope,
                                    reinterpret_cast<__acpp_int16 *>(addr),
                                    __builtin_bit_cast(__acpp_int8, x));
  } else if constexpr (sizeof(T) == 4) {
    __acpp_sscp_atomic_store_i32(S, order, scope,
                                    reinterpret_cast<__acpp_int32 *>(addr),
                                    __builtin_bit_cast(__acpp_int32, x));
  } else if constexpr (sizeof(T) == 8) {
    __acpp_sscp_atomic_store_i64(S, order, scope,
                                    reinterpret_cast<__acpp_int64 *>(addr),
                                    __builtin_bit_cast(__acpp_int64, x));
  }
}



template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {

  if constexpr (sizeof(T) == 1) {
    return_cast(__acpp_sscp_atomic_load_i8(
        S, order, scope, reinterpret_cast<__acpp_int8 *>(addr)));
  } else if constexpr (sizeof(T) == 2) {
    return_cast(__acpp_sscp_atomic_load_i16(
        S, order, scope, reinterpret_cast<__acpp_int16 *>(addr)));
  } else if constexpr (sizeof(T) == 4) {
    return_cast(__acpp_sscp_atomic_load_i32(
        S, order, scope, reinterpret_cast<__acpp_int32 *>(addr)));
  } else if constexpr (sizeof(T) == 8) {
    return_cast(__acpp_sscp_atomic_load_i64(
        S, order, scope, reinterpret_cast<__acpp_int64 *>(addr)));
  }
  __builtin_unreachable();
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    return_cast(__acpp_sscp_atomic_exchange_i8(
        S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
        __builtin_bit_cast(__acpp_int8, x)));
  } else if constexpr (sizeof(T) == 2) {
    return_cast(__acpp_sscp_atomic_exchange_i16(
        S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
        __builtin_bit_cast(__acpp_int16, x)));
  } else if constexpr (sizeof(T) == 4) {
    return_cast(__acpp_sscp_atomic_exchange_i32(
        S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
        __builtin_bit_cast(__acpp_int32, x)));
  } else if constexpr (sizeof(T) == 8) {
    return_cast(__acpp_sscp_atomic_exchange_i64(
        S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
        __builtin_bit_cast(__acpp_int64, x)));
  }
  __builtin_unreachable();
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    return __acpp_sscp_cmp_exch_weak_i8(
        S, success, failure, scope, reinterpret_cast<__acpp_int8 *>(addr),
        reinterpret_cast<__acpp_int8 *>(&expected),
        __builtin_bit_cast(__acpp_int8, desired));
  } else if constexpr (sizeof(T) == 2) {
    return __acpp_sscp_cmp_exch_weak_i16(
        S, success, failure, scope, reinterpret_cast<__acpp_int16 *>(addr),
        reinterpret_cast<__acpp_int16 *>(&expected),
        __builtin_bit_cast(__acpp_int16, desired));
  } else if constexpr (sizeof(T) == 4) {
    return __acpp_sscp_cmp_exch_weak_i32(
        S, success, failure, scope, reinterpret_cast<__acpp_int32 *>(addr),
        reinterpret_cast<__acpp_int32 *>(&expected),
        __builtin_bit_cast(__acpp_int32, desired));
  } else if constexpr (sizeof(T) == 8) {
    return __acpp_sscp_cmp_exch_weak_i64(
        S, success, failure, scope, reinterpret_cast<__acpp_int64 *>(addr),
        reinterpret_cast<__acpp_int64 *>(&expected),
        __builtin_bit_cast(__acpp_int64, desired));
  }
  __builtin_unreachable();
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  if constexpr (sizeof(T) == 1) {
    return __acpp_sscp_cmp_exch_strong_i8(
        S, success, failure, scope, reinterpret_cast<__acpp_int8 *>(addr),
        reinterpret_cast<__acpp_int8 *>(&expected),
        __builtin_bit_cast(__acpp_int8, desired));
  } else if constexpr (sizeof(T) == 2) {
    return __acpp_sscp_cmp_exch_strong_i16(
        S, success, failure, scope, reinterpret_cast<__acpp_int16 *>(addr),
        reinterpret_cast<__acpp_int16 *>(&expected),
        __builtin_bit_cast(__acpp_int16, desired));
  } else if constexpr (sizeof(T) == 4) {
    return __acpp_sscp_cmp_exch_strong_i32(
        S, success, failure, scope, reinterpret_cast<__acpp_int32 *>(addr),
        reinterpret_cast<__acpp_int32 *>(&expected),
        __builtin_bit_cast(__acpp_int32, desired));
  } else if constexpr (sizeof(T) == 8) {
    return __acpp_sscp_cmp_exch_strong_i64(
        S, success, failure, scope, reinterpret_cast<__acpp_int64 *>(addr),
        reinterpret_cast<__acpp_int64 *>(&expected),
        __builtin_bit_cast(__acpp_int64, desired));
  }
  __builtin_unreachable();
}

// Integral values only

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    return_cast(__acpp_sscp_atomic_fetch_and_i8(
        S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
        __builtin_bit_cast(__acpp_int8, x)));
  } else if constexpr (sizeof(T) == 2) {
    return_cast(__acpp_sscp_atomic_fetch_and_i16(
        S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
        __builtin_bit_cast(__acpp_int16, x)));
  } else if constexpr (sizeof(T) == 4) {
    return_cast(__acpp_sscp_atomic_fetch_and_i32(
        S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
        __builtin_bit_cast(__acpp_int32, x)));
  } else if constexpr (sizeof(T) == 8) {
    return_cast(__acpp_sscp_atomic_fetch_and_i64(
        S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
        __builtin_bit_cast(__acpp_int64, x)));
  }
  __builtin_unreachable();
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    return_cast(__acpp_sscp_atomic_fetch_or_i8(
        S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
        __builtin_bit_cast(__acpp_int8, x)));
  } else if constexpr (sizeof(T) == 2) {
    return_cast(__acpp_sscp_atomic_fetch_or_i16(
        S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
        __builtin_bit_cast(__acpp_int16, x)));
  } else if constexpr (sizeof(T) == 4) {
    return_cast(__acpp_sscp_atomic_fetch_or_i32(
        S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
        __builtin_bit_cast(__acpp_int32, x)));
  } else if constexpr (sizeof(T) == 8) {
    return_cast(__acpp_sscp_atomic_fetch_or_i64(
        S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
        __builtin_bit_cast(__acpp_int64, x)));
  }
  __builtin_unreachable();
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr (sizeof(T) == 1) {
    return_cast(__acpp_sscp_atomic_fetch_xor_i8(
        S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
        __builtin_bit_cast(__acpp_int8, x)));
  } else if constexpr (sizeof(T) == 2) {
    return_cast(__acpp_sscp_atomic_fetch_xor_i16(
        S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
        __builtin_bit_cast(__acpp_int16, x)));
  } else if constexpr (sizeof(T) == 4) {
    return_cast(__acpp_sscp_atomic_fetch_xor_i32(
        S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
        __builtin_bit_cast(__acpp_int32, x)));
  } else if constexpr (sizeof(T) == 8) {
    return_cast(__acpp_sscp_atomic_fetch_xor_i64(
        S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
        __builtin_bit_cast(__acpp_int64, x)));
  }
  __builtin_unreachable();
}

// Floating point and integral values

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr(std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_add_i8(
          S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
          __builtin_bit_cast(__acpp_int8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_add_i16(
          S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
          __builtin_bit_cast(__acpp_int16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_add_i32(
          S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
          __builtin_bit_cast(__acpp_int32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_add_i64(
          S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
          __builtin_bit_cast(__acpp_int64, x)));
    }
  } else {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_add_u8(
          S, order, scope, reinterpret_cast<__acpp_uint8 *>(addr),
          __builtin_bit_cast(__acpp_uint8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_add_u16(
          S, order, scope, reinterpret_cast<__acpp_uint16 *>(addr),
          __builtin_bit_cast(__acpp_uint16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_add_u32(
          S, order, scope, reinterpret_cast<__acpp_uint32 *>(addr),
          __builtin_bit_cast(__acpp_uint32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_add_u64(
          S, order, scope, reinterpret_cast<__acpp_uint64 *>(addr),
          __builtin_bit_cast(__acpp_uint64, x)));
    }
  }
  __builtin_unreachable();
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_add(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_add_f32(S, order, scope, addr, x);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_add(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_add_f64(S, order, scope, addr, x);
}




template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr(std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_sub_i8(
          S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
          __builtin_bit_cast(__acpp_int8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_sub_i16(
          S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
          __builtin_bit_cast(__acpp_int16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_sub_i32(
          S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
          __builtin_bit_cast(__acpp_int32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_sub_i64(
          S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
          __builtin_bit_cast(__acpp_int64, x)));
    }
  } else {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_sub_u8(
          S, order, scope, reinterpret_cast<__acpp_uint8 *>(addr),
          __builtin_bit_cast(__acpp_uint8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_sub_u16(
          S, order, scope, reinterpret_cast<__acpp_uint16 *>(addr),
          __builtin_bit_cast(__acpp_uint16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_sub_u32(
          S, order, scope, reinterpret_cast<__acpp_uint32 *>(addr),
          __builtin_bit_cast(__acpp_uint32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_sub_u64(
          S, order, scope, reinterpret_cast<__acpp_uint64 *>(addr),
          __builtin_bit_cast(__acpp_uint64, x)));
    }
  }
  __builtin_unreachable();
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_sub(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_sub_f32(S, order, scope, addr, x);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_sub(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_sub_f64(S, order, scope, addr, x);
}





template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr(std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_min_i8(
          S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
          __builtin_bit_cast(__acpp_int8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_min_i16(
          S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
          __builtin_bit_cast(__acpp_int16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_min_i32(
          S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
          __builtin_bit_cast(__acpp_int32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_min_i64(
          S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
          __builtin_bit_cast(__acpp_int64, x)));
    }
  } else {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_min_u8(
          S, order, scope, reinterpret_cast<__acpp_uint8 *>(addr),
          __builtin_bit_cast(__acpp_uint8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_min_u16(
          S, order, scope, reinterpret_cast<__acpp_uint16 *>(addr),
          __builtin_bit_cast(__acpp_uint16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_min_u32(
          S, order, scope, reinterpret_cast<__acpp_uint32 *>(addr),
          __builtin_bit_cast(__acpp_uint32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_min_u64(
          S, order, scope, reinterpret_cast<__acpp_uint64 *>(addr),
          __builtin_bit_cast(__acpp_uint64, x)));
    }
  }
  __builtin_unreachable();
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_min(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_min_f32(S, order, scope, addr, x);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_min(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_min_f64(S, order, scope, addr, x);
}






template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  if constexpr(std::is_signed_v<T>) {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_max_i8(
          S, order, scope, reinterpret_cast<__acpp_int8 *>(addr),
          __builtin_bit_cast(__acpp_int8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_max_i16(
          S, order, scope, reinterpret_cast<__acpp_int16 *>(addr),
          __builtin_bit_cast(__acpp_int16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_max_i32(
          S, order, scope, reinterpret_cast<__acpp_int32 *>(addr),
          __builtin_bit_cast(__acpp_int32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_max_i64(
          S, order, scope, reinterpret_cast<__acpp_int64 *>(addr),
          __builtin_bit_cast(__acpp_int64, x)));
    }
  } else {
    if constexpr (sizeof(T) == 1) {
      return_cast(__acpp_sscp_atomic_fetch_max_u8(
          S, order, scope, reinterpret_cast<__acpp_uint8 *>(addr),
          __builtin_bit_cast(__acpp_uint8, x)));
    } else if constexpr (sizeof(T) == 2) {
      return_cast(__acpp_sscp_atomic_fetch_max_u16(
          S, order, scope, reinterpret_cast<__acpp_uint16 *>(addr),
          __builtin_bit_cast(__acpp_uint16, x)));
    } else if constexpr (sizeof(T) == 4) {
      return_cast(__acpp_sscp_atomic_fetch_max_u32(
          S, order, scope, reinterpret_cast<__acpp_uint32 *>(addr),
          __builtin_bit_cast(__acpp_uint32, x)));
    } else if constexpr (sizeof(T) == 8) {
      return_cast(__acpp_sscp_atomic_fetch_max_u64(
          S, order, scope, reinterpret_cast<__acpp_uint64 *>(addr),
          __builtin_bit_cast(__acpp_uint64, x)));
    }
  }
  __builtin_unreachable();
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_max(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_max_f32(S, order, scope, addr, x);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_max(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  return __acpp_sscp_atomic_fetch_max_f64(S, order, scope, addr, x);
}



#undef return_cast

}
}
}

#endif

#endif
