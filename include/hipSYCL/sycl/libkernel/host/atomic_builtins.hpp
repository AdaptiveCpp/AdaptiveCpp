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
#ifndef HIPSYCL_ATOMIC_HOST_BUILTINS_HPP
#define HIPSYCL_ATOMIC_HOST_BUILTINS_HPP

#include <cstdint>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/bit_cast.hpp"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST

namespace hipsycl {
namespace sycl {
namespace detail::host_builtins {

inline constexpr int builtin_memory_order(memory_order o) noexcept {
  switch(o){
    case memory_order::relaxed:
      return __ATOMIC_RELAXED;
    case memory_order::acquire:
      return __ATOMIC_ACQUIRE;
    case memory_order::release:
      return __ATOMIC_RELEASE;
    case memory_order::acq_rel:
      return __ATOMIC_ACQ_REL;
    case memory_order::seq_cst:
      return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_RELAXED;
}

HIPSYCL_BUILTIN int32_t float_as_int(float f) noexcept {
  return bit_cast<int32_t>(f);
}

HIPSYCL_BUILTIN int64_t float_as_int(double f) noexcept {
  return bit_cast<int64_t>(f);
}

HIPSYCL_BUILTIN float int_as_float(int32_t i) noexcept {
  return bit_cast<float>(i);
}

HIPSYCL_BUILTIN double int_as_float(int64_t i) noexcept {
  return bit_cast<double>(i);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN void __acpp_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  __atomic_store_n(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN void __acpp_atomic_store(float* addr, float x, memory_order order,
                                            memory_scope scope) noexcept {
  __acpp_atomic_store<S>(reinterpret_cast<int32_t *>(addr), float_as_int(x),
                            order, scope);
}

template <access::address_space S>
HIPSYCL_BUILTIN void __acpp_atomic_store(double* addr, double x, memory_order order,
                                            memory_scope scope) noexcept {
  __acpp_atomic_store<S>(reinterpret_cast<int64_t *>(addr), float_as_int(x),
                            order, scope);
}



template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {
  return __atomic_load_n(addr, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_load(float* addr, memory_order order,
                                            memory_scope scope) noexcept {
  int32_t v =
      __acpp_atomic_load<S>(reinterpret_cast<int32_t *>(addr), order, scope);

  return int_as_float(v);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_load(double* addr, memory_order order,
                                            memory_scope scope) noexcept {
  int64_t v = __acpp_atomic_load<S>(
      reinterpret_cast<int64_t *>(addr), order, scope);
  
  return int_as_float(v);
}



template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  return __atomic_exchange_n(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_exchange(
    float *addr, float x, memory_order order, memory_scope scope) noexcept {
  
  int32_t v = __acpp_atomic_exchange<S>(reinterpret_cast<int32_t *>(addr),
                                           float_as_int(x), order, scope);
  return int_as_float(v);
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_exchange(
    double *addr, double x, memory_order order, memory_scope scope) noexcept {
  
  int64_t v = __acpp_atomic_exchange<S>(reinterpret_cast<int64_t *>(addr),
                                           float_as_int(x), order, scope);
  return int_as_float(v);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  return __atomic_compare_exchange_n(addr, &expected, desired, true,
                                     builtin_memory_order(success),
                                     builtin_memory_order(failure));
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    float *addr, float &expected, float desired, memory_order success,
    memory_order failure, memory_scope scope) noexcept {
  
  int32_t expected_int = float_as_int(expected);
  int32_t desired_int = float_as_int(desired);
  
  bool res = __acpp_atomic_compare_exchange_weak<S>(
      reinterpret_cast<int32_t *>(addr), expected_int, desired_int, success,
      failure, scope);
  
  expected = int_as_float(expected_int);
  return res;
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    double *addr, double &expected, double desired, memory_order success,
    memory_order failure, memory_scope scope) noexcept {
  
  int64_t expected_int = float_as_int(expected);
  int64_t desired_int = float_as_int(desired);
  
  bool res = __acpp_atomic_compare_exchange_weak<S>(
      reinterpret_cast<int64_t *>(addr), expected_int, desired_int, success,
      failure, scope);
  
  expected = int_as_float(expected_int);
  return res;
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  return __atomic_compare_exchange_n(addr, &expected, desired, false,
                                     builtin_memory_order(success),
                                     builtin_memory_order(failure));
}


template <access::address_space S>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    float *addr, float &expected, float desired, memory_order success,
    memory_order failure, memory_scope scope) noexcept {
  
  int32_t expected_int = float_as_int(expected);
  int32_t desired_int = float_as_int(desired);
  
  bool res = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<int32_t *>(addr), expected_int, desired_int, success,
      failure, scope);
  
  expected = int_as_float(expected_int);
  return res;
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    double *addr, double &expected, double desired, memory_order success,
    memory_order failure, memory_scope scope) noexcept {
  
  int64_t expected_int = float_as_int(expected);
  int64_t desired_int = float_as_int(desired);
  
  bool res = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<int64_t *>(addr), expected_int, desired_int, success,
      failure, scope);
  
  expected = int_as_float(expected_int);
  return res;
}

// Integral values only

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return __atomic_fetch_and(addr, x, builtin_memory_order(order));
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return __atomic_fetch_or(addr, x, builtin_memory_order(order));
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return __atomic_fetch_xor(addr, x, builtin_memory_order(order));
}

// Floating point and integral values

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return __atomic_fetch_add(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_add(
    float *addr, float x, memory_order order, memory_scope scope) noexcept {
  
  float old = __acpp_atomic_load<S>(addr, order, scope);
  while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, old + x, order,
                                                      order, scope))
    ;
  return old;
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_add(
    double *addr, double x, memory_order order, memory_scope scope) noexcept {
  
  double old = __acpp_atomic_load<S>(addr, order, scope);
  while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, old + x, order,
                                                      order, scope))
    ;
  return old;
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return __atomic_fetch_sub(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __acpp_atomic_fetch_sub(
    float *addr, float x, memory_order order, memory_scope scope) noexcept {
  
  float old = __acpp_atomic_load<S>(addr, order, scope);
  while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, old - x, order,
                                                      order, scope))
    ;
  return old;
}

template <access::address_space S>
HIPSYCL_BUILTIN double __acpp_atomic_fetch_sub(
    double *addr, double x, memory_order order, memory_scope scope) noexcept {
  
  double old = __acpp_atomic_load<S>(addr, order, scope);
  while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, old - x, order,
                                                      order, scope))
    ;
  return old;
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  T old = __acpp_atomic_load<S>(addr, order, scope);
  do{
    if (old < x) return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, x, order,
                                                           order, scope));
  return x;
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  T old = __acpp_atomic_load<S>(addr, order, scope);
  do{
    if (old > x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(addr, old, x, order,
                                                        order, scope));
  return x;
}

}
}
}
#endif

#endif
