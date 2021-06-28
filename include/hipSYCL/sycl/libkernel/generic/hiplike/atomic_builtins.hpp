/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP
#define HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/detail/util.hpp"

#include <type_traits>

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl {
namespace detail {

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

template <access::address_space S, class T>
HIPSYCL_BUILTIN void __hipsycl_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  __atomic_store_n(addr, x, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __hipsycl_atomic_store(float *addr, float x, memory_order order,
                                            memory_scope scope) noexcept {
  __atomic_store_n(reinterpret_cast<unsigned int *>(addr),
                   bit_cast<unsigned int>(x), builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN double __hipsycl_atomic_store(double *addr, double x, memory_order order,
                                            memory_scope scope) noexcept {
  __atomic_store_n(reinterpret_cast<unsigned long long *>(addr),
                   bit_cast<unsigned long long>(x), builtin_memory_order(order));
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {
  return __atomic_load_n(addr, builtin_memory_order(order));
}

template <access::address_space S>
HIPSYCL_BUILTIN float __hipsycl_atomic_load(float *addr, memory_order order,
                                            memory_scope scope) noexcept {
  return bit_cast<float>(__atomic_load_n(reinterpret_cast<unsigned int *>(addr),
                                         builtin_memory_order(order)));
}

template <access::address_space S>
HIPSYCL_BUILTIN double __hipsycl_atomic_load(double *addr, memory_order order,
                                             memory_scope scope) noexcept {
  return bit_cast<double>(
      __atomic_load_n(reinterpret_cast<unsigned long long *>(addr),
                      builtin_memory_order(order)));
}

//********************* atomic exchange ***********************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  return atomicExch(addr, x);
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_BUILTIN T __hipsycl_bitcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return bit_cast<T>(__hipsycl_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), bit_cast<InvokedT>(x), order, scope));
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_BUILTIN T __hipsycl_staticcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return static_cast<T>(__hipsycl_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), static_cast<InvokedT>(x), order, scope));
}


// No direct support for long long, long and unsigned long, and double
template <access::address_space S>
HIPSYCL_BUILTIN long long __hipsycl_atomic_exchange(long long *addr, long long x,
                                            memory_order order,
                                            memory_scope scope) noexcept {

  return __hipsycl_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

template <access::address_space S>
HIPSYCL_BUILTIN long __hipsycl_atomic_exchange(long *addr, long x,
                                            memory_order order,
                                            memory_scope scope) noexcept {
  if constexpr(sizeof(long) == 4) {
    return __hipsycl_staticcast_atomic_exchange<S, int>(addr, x, order, scope);
  } else {
    return __hipsycl_bitcast_atomic_exchange<S, unsigned long long>(addr, x, order,
                                                                    scope);
  }
}

template <access::address_space S>
HIPSYCL_BUILTIN unsigned long
__hipsycl_atomic_exchange(unsigned long *addr, unsigned long x,
                          memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return __hipsycl_staticcast_atomic_exchange<S, unsigned int>(
        addr, x, order, scope);
  } else {
    return __hipsycl_staticcast_atomic_exchange<S, unsigned long long>(
        addr, x, order, scope);
  }
}

template <access::address_space S>
HIPSYCL_BUILTIN double __hipsycl_atomic_exchange(double *addr, double x,
                                            memory_order order,
                                            memory_scope scope) noexcept {
  return __hipsycl_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

// ******************* atomic compare exchange strong ********************

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  T old = expected;
  expected = atomicCAS(addr, expected, desired);
  return old == expected;
}


template <access::address_space S, class InvokedT, class T>
HIPSYCL_BUILTIN bool __hipsycl_bitcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = bit_cast<InvokedT>(expected);
  
  bool r = __hipsycl_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      bit_cast<InvokedT>(desired), success, failure, scope);

  expected = bit_cast<T>(cast_expected);
  return r;
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_BUILTIN bool __hipsycl_staticcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = static_cast<InvokedT>(expected);
  
  bool r = __hipsycl_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      static_cast<InvokedT>(desired), success, failure, scope);

  expected = static_cast<T>(cast_expected);
  return r;
}

// No direct support for long long, long and unsigned long, float, double

template <access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    long long *addr, long long &expected, long long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __hipsycl_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    long *addr, long &expected, long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr (sizeof(long) == 4) {
    return __hipsycl_staticcast_compare_exchange_strong<S, int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __hipsycl_bitcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    unsigned long *addr, unsigned long &expected, unsigned long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr(sizeof(unsigned long) == 4) {
    return __hipsycl_staticcast_compare_exchange_strong<S, unsigned int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __hipsycl_staticcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    float *addr, float &expected, float desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __hipsycl_bitcast_compare_exchange_strong<S, int>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    double *addr, double &expected, double desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __hipsycl_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

// ******************* atomic compare exchange weak ********************

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  return __hipsycl_atomic_compare_exchange_strong<S>(
      addr, expected, desired, success, failure, scope);
}

// *********************** Integral values only *****************************

// Defines overloads for integral atomics for types that are not natively
// supported using bitcasts: long long, long and unsigned long
#define HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(            \
    name, bitcast_name)                                                        \
  template <access::address_space S, class InvokedT, class T>                  \
  HIPSYCL_BUILTIN T bitcast_name(T *addr, T x, memory_order order,             \
                                 memory_scope scope) noexcept {                \
    return bit_cast<T>(name<S>(reinterpret_cast<InvokedT *>(addr),             \
                               bit_cast<InvokedT>(x), order, scope));          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_BUILTIN long long name(long long *addr, long long x,                 \
                                 memory_order order,                           \
                                 memory_scope scope) noexcept {                \
    return bitcast_name<S, unsigned long long>(addr, x, order, scope);         \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_BUILTIN long name(long *addr, long x, memory_order order,            \
                            memory_scope scope) noexcept {                     \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, int>(addr, x, order, scope);                      \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_BUILTIN unsigned long name(unsigned long *addr, unsigned long x,     \
                                     memory_order order,                       \
                                     memory_scope scope) noexcept {            \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, unsigned int>(addr, x, order, scope);             \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }

// ******************** atomic fetch and ************************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicAnd(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_and, __hipsycl_bitcast_atomic_fetch_and);

// ************************ atomic fetch or **********************


template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicOr(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_or, __hipsycl_bitcast_atomic_fetch_or);

// ************************ atomic fetch xor **********************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicXor(addr, x);
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_xor, __hipsycl_bitcast_atomic_fetch_xor);

// ***************** Floating point and integral values *************


// *********************** atomic add ****************************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicAdd(addr, x);
}

// atomicAdd supports float and double, so we only need to complete the integral
// overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_add, __hipsycl_bitcast_atomic_fetch_add);

// *********************** atomic sub ****************************

// Only supports int and unsigned
template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicSub(addr, x);
}

// Need CAS loop for 64bit types - the definition of this
// is required in order to be able to use the 
// HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS macro

#define HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(Type)                   \
  template <access::address_space S>                                           \
  HIPSYCL_BUILTIN Type __hipsycl_atomic_fetch_sub(                             \
      Type *addr, Type x, memory_order order, memory_scope scope) noexcept {   \
    Type old = __hipsycl_atomic_load<S>(addr, order, scope);                   \
    while (!__hipsycl_atomic_compare_exchange_strong<S>(addr, old, old - x,    \
                                                        order, order, scope))  \
      ;                                                                        \
    return old;                                                                \
  }

HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(unsigned long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(float)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(double)

template <access::address_space S>
HIPSYCL_BUILTIN long __hipsycl_atomic_fetch_sub(long *addr, long x,
                                                memory_order order,
                                                memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return static_cast<long>(__hipsycl_atomic_fetch_sub<S>(
        reinterpret_cast<int *>(addr), static_cast<int>(x), order, scope));
  } else {
    return static_cast<long>(__hipsycl_atomic_fetch_sub<S>(
        reinterpret_cast<long long *>(addr), static_cast<long long>(x), order,
        scope));
  }
}

template <access::address_space S>
HIPSYCL_BUILTIN unsigned long
__hipsycl_atomic_fetch_sub(unsigned long *addr, unsigned long x,
                           memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(unsigned long) == 4) {
    return static_cast<unsigned long>(__hipsycl_atomic_fetch_sub<S>(
        reinterpret_cast<unsigned *>(addr), static_cast<unsigned>(x), order,
        scope));
  } else {
    return static_cast<unsigned long>(
        __hipsycl_atomic_fetch_sub<S>(
            reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(x), order, scope));
  }
}

// *********************** atomic min ****************************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicMin(addr, x);
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_min, __hipsycl_bitcast_atomic_fetch_min);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_BUILTIN float __hipsycl_atomic_fetch_min(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  float old = __hipsycl_atomic_load<S>(addr, order, scope);
  do {
    if (old < x)
      return old;
  } while (!__hipsycl_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

template <access::address_space S>
HIPSYCL_BUILTIN double __hipsycl_atomic_fetch_min(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  double old = __hipsycl_atomic_load<S>(addr, order, scope);
  do {
    if (old < x)
      return old;
  } while (!__hipsycl_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

// *********************** atomic max ****************************

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicMax(addr, x);
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS(
    __hipsycl_atomic_fetch_max, __hipsycl_bitcast_atomic_fetch_max);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_BUILTIN float __hipsycl_atomic_fetch_max(float *addr, float x,
                                                 memory_order order,
                                                 memory_scope scope) noexcept {
  float old = __hipsycl_atomic_load<S>(addr, order, scope);
  do {
    if (old > x)
      return old;
  } while (!__hipsycl_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

template <access::address_space S>
HIPSYCL_BUILTIN double __hipsycl_atomic_fetch_max(double *addr, double x,
                                                  memory_order order,
                                                  memory_scope scope) noexcept {
  double old = __hipsycl_atomic_load<S>(addr, order, scope);
  do {
    if (old > x)
      return old;
  } while (!__hipsycl_atomic_compare_exchange_strong<S>(
      addr, old, x, order, order, scope));
  return x;
}

}
}
}

#endif

#endif
