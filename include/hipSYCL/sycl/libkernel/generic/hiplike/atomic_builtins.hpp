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
#ifndef HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP
#define HIPSYCL_ATOMIC_HIPLIKE_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "hipSYCL/sycl/libkernel/bit_cast.hpp"

#include <type_traits>


#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                   \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP

namespace hipsycl {
namespace sycl {
namespace detail::hiplike_builtins {

inline constexpr int builtin_memory_order(memory_order o) noexcept {
  switch(o) {
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

inline constexpr memory_order
load_order_for_rmw(memory_order order) noexcept {
  if(order == memory_order::release)
    return memory_order::relaxed;
  if(order == memory_order::acq_rel)
    return memory_order::acquire;
  return order;
}

inline constexpr memory_order
failure_order_for_rmw(memory_order order) noexcept {
  if(order == memory_order::release)
    return memory_order::relaxed;
  if(order == memory_order::acq_rel)
    return memory_order::acquire;
  return order;
}

#if !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
inline constexpr int builtin_memory_scope(memory_scope s) noexcept {
  switch(s) {
  case memory_scope::work_item:
    return __MEMORY_SCOPE_SINGLE;
  case memory_scope::sub_group:
    return __MEMORY_SCOPE_WVFRNT;
  case memory_scope::work_group:
    return __MEMORY_SCOPE_WRKGRP;
  case memory_scope::device:
    return __MEMORY_SCOPE_DEVICE;
  case memory_scope::system:
    return __MEMORY_SCOPE_SYSTEM;
  }
  return __MEMORY_SCOPE_SYSTEM;
}
#endif

HIPSYCL_HIPLIKE_BUILTIN
void __acpp_atomic_fence_impl(memory_scope scope) noexcept {
  if(scope == memory_scope::system)
    __threadfence_system();
  else if(scope == memory_scope::device)
    __threadfence();
  else
    // Atomic operations do not accept work-item scope. Using a work-group
    // fence for sub-group scope is conservative and therefore valid.
    __threadfence_block();
}

HIPSYCL_HIPLIKE_BUILTIN
void __acpp_atomic_pre_fence(memory_order order, memory_scope scope) noexcept {
  if(order == memory_order::release || order == memory_order::acq_rel ||
     order == memory_order::seq_cst)
    __acpp_atomic_fence_impl(scope);
}

HIPSYCL_HIPLIKE_BUILTIN
void __acpp_atomic_post_fence(memory_order order, memory_scope scope) noexcept {
  if(order == memory_order::acquire || order == memory_order::acq_rel ||
     order == memory_order::seq_cst)
    __acpp_atomic_fence_impl(scope);
}

#ifdef ACPP_LIBKERNEL_CUDA_NVCXX

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
#define ACPP_NVCXX_SYSTEM_ATOMIC(op, addr, x) op##_system(addr, x)
#else
#define ACPP_NVCXX_SYSTEM_ATOMIC(op, addr, x) op(addr, x)
#endif

#define ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(helper, op)                    \
  template<class T>                                                            \
  HIPSYCL_HIPLIKE_BUILTIN T helper(T* addr, T x, memory_scope scope) noexcept {\
    if(scope == memory_scope::system)                                          \
      return ACPP_NVCXX_SYSTEM_ATOMIC(op, addr, x);                            \
    if(scope == memory_scope::work_group || scope == memory_scope::sub_group)  \
      return op##_block(addr, x);                                              \
    return op(addr, x);                                                        \
  }

ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_exchange, atomicExch)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_add, atomicAdd)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_sub, atomicSub)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_and, atomicAnd)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_or, atomicOr)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_xor, atomicXor)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_min, atomicMin)
ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC(__acpp_nvcxx_fetch_max, atomicMax)

#undef ACPP_DEFINE_NVCXX_SCOPED_BINARY_ATOMIC
#undef ACPP_NVCXX_SYSTEM_ATOMIC

template<class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_nvcxx_compare_exchange(
    T* addr, T expected, T desired, memory_scope scope) noexcept {
  if(scope == memory_scope::system) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
    return atomicCAS_system(addr, expected, desired);
#else
    return atomicCAS(addr, expected, desired);
#endif
  }
  if(scope == memory_scope::work_group || scope == memory_scope::sub_group)
    return atomicCAS_block(addr, expected, desired);
  return atomicCAS(addr, expected, desired);
}

template<class Word>
HIPSYCL_HIPLIKE_BUILTIN Word __acpp_nvcxx_atomic_load_word(
    Word* addr, memory_order order, memory_scope scope) noexcept {
  const Word zero = static_cast<Word>(0);
  const Word result = __acpp_nvcxx_compare_exchange(addr, zero, zero, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
}

template<class Word>
HIPSYCL_HIPLIKE_BUILTIN void __acpp_nvcxx_atomic_store_word(
    Word* addr, Word x, memory_order order, memory_scope scope) noexcept {
  __acpp_atomic_pre_fence(order, scope);
  (void)__acpp_nvcxx_exchange(addr, x, scope);
}

#endif

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(T *addr, T x, memory_order order,
                    memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  __scoped_atomic_store_n(addr, x, builtin_memory_order(order),
                          builtin_memory_scope(scope));
#else
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "Unsupported CUDA atomic storage width");
  if constexpr(sizeof(T) == 4) {
    auto* p = reinterpret_cast<unsigned int*>(addr);
    __acpp_nvcxx_atomic_store_word(
        p, bit_cast<unsigned int>(x), order, scope);
  } else {
    auto* p = reinterpret_cast<unsigned long long*>(addr);
    __acpp_nvcxx_atomic_store_word(
        p, bit_cast<unsigned long long>(x), order, scope);
  }
#endif
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(float *addr, float x, memory_order order,
                    memory_scope scope) noexcept {
  __acpp_atomic_store<S>(
      reinterpret_cast<unsigned int*>(addr), bit_cast<unsigned int>(x),
      order, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN void
__acpp_atomic_store(double *addr, double x, memory_order order,
                    memory_scope scope) noexcept {
  __acpp_atomic_store<S>(
      reinterpret_cast<unsigned long long*>(addr),
      bit_cast<unsigned long long>(x), order, scope);
}

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T
__acpp_atomic_load(T *addr, memory_order order,
                   memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_load_n(addr, builtin_memory_order(order),
                                builtin_memory_scope(scope));
#else
  static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                "Unsupported CUDA atomic storage width");
  if constexpr(sizeof(T) == 4) {
    const auto result = __acpp_nvcxx_atomic_load_word(
        reinterpret_cast<unsigned int*>(addr), order, scope);
    return bit_cast<T>(result);
  } else {
    const auto result = __acpp_nvcxx_atomic_load_word(
        reinterpret_cast<unsigned long long*>(addr), order, scope);
    return bit_cast<T>(result);
  }
#endif
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_load(float *addr, memory_order order,
                   memory_scope scope) noexcept {
  return bit_cast<float>(__acpp_atomic_load<S>(
      reinterpret_cast<unsigned int*>(addr), order, scope));
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_load(double *addr, memory_order order,
                   memory_scope scope) noexcept {
  return bit_cast<double>(__acpp_atomic_load<S>(
      reinterpret_cast<unsigned long long*>(addr), order, scope));
}

//********************* atomic exchange ***********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_exchange_n(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_exchange(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_bitcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return bit_cast<T>(__acpp_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), bit_cast<InvokedT>(x), order, scope));
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_staticcast_atomic_exchange(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {

  return static_cast<T>(__acpp_atomic_exchange<S>(
      reinterpret_cast<InvokedT*>(addr), static_cast<InvokedT>(x), order, scope));
}


// No direct support for long long, long and unsigned long, and double
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long long
__acpp_atomic_exchange(long long *addr, long long x, memory_order order,
                          memory_scope scope) noexcept {

  return __acpp_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long
__acpp_atomic_exchange(long *addr, long x, memory_order order,
                          memory_scope scope) noexcept {
  if constexpr(sizeof(long) == 4) {
    return __acpp_staticcast_atomic_exchange<S, int>(addr, x, order, scope);
  } else {
    return __acpp_bitcast_atomic_exchange<S, unsigned long long>(addr, x, order,
                                                                    scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN unsigned long
__acpp_atomic_exchange(unsigned long *addr, unsigned long x,
                          memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return __acpp_staticcast_atomic_exchange<S, unsigned int>(
        addr, x, order, scope);
  } else {
    return __acpp_staticcast_atomic_exchange<S, unsigned long long>(
        addr, x, order, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_exchange(double *addr, double x, memory_order order,
                          memory_scope scope) noexcept {
  return __acpp_bitcast_atomic_exchange<S, unsigned long long>(
      addr, x, order, scope);
}

// ******************* atomic compare exchange strong ********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_compare_exchange_n(
      addr, &expected, desired, false, builtin_memory_order(success),
      builtin_memory_order(failure), builtin_memory_scope(scope));
#else
  // A release fence before the operation is conservative on a failed CAS,
  // while preserving the required successful release semantics.
  __acpp_atomic_pre_fence(success, scope);
  const T old_expected = expected;
  expected = __acpp_nvcxx_compare_exchange(addr, expected, desired, scope);
  const bool succeeded = old_expected == expected;
  __acpp_atomic_post_fence(succeeded ? success : failure, scope);
  return succeeded;
#endif
}


template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_bitcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = bit_cast<InvokedT>(expected);
  
  bool r = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      bit_cast<InvokedT>(desired), success, failure, scope);

  expected = bit_cast<T>(cast_expected);
  return r;
}

template <access::address_space S, class InvokedT, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_staticcast_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  InvokedT cast_expected = static_cast<InvokedT>(expected);
  
  bool r = __acpp_atomic_compare_exchange_strong<S>(
      reinterpret_cast<InvokedT *>(addr), cast_expected,
      static_cast<InvokedT>(desired), success, failure, scope);

  expected = static_cast<T>(cast_expected);
  return r;
}

// No direct support for long long, long and unsigned long, float, double

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    long long *addr, long long &expected, long long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    long *addr, long &expected, long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr (sizeof(long) == 4) {
    return __acpp_staticcast_compare_exchange_strong<S, int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    unsigned long *addr, unsigned long &expected, unsigned long desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  if constexpr(sizeof(unsigned long) == 4) {
    return __acpp_staticcast_compare_exchange_strong<S, unsigned int>(
        addr, expected, desired, success, failure, scope);
  } else {
    return __acpp_staticcast_compare_exchange_strong<S, unsigned long long>(
        addr, expected, desired, success, failure, scope);
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    float *addr, float &expected, float desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, int>(
      addr, expected, desired, success, failure, scope);
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    double *addr, double &expected, double desired,
    memory_order success, memory_order failure, memory_scope scope) noexcept {

  return __acpp_bitcast_compare_exchange_strong<S, unsigned long long>(
      addr, expected, desired, success, failure, scope);
}

// ******************* atomic compare exchange weak ********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  return __acpp_atomic_compare_exchange_strong<S>(
      addr, expected, desired, success, failure, scope);
}

// *********************** Integral values only *****************************

// Defines overloads for integral atomics for types that are not natively
// supported using bitcasts (long long, long and unsigned long), for operations
// that are agnostic to the sign of the operands (e.g. atomicAdd).
#define HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_AGNOSTIC(   \
    name, bitcast_name)                                                        \
  template <access::address_space S, class InvokedT, class T>                  \
  HIPSYCL_HIPLIKE_BUILTIN T bitcast_name(T *addr, T x, memory_order order,     \
                                         memory_scope scope) noexcept {        \
    return bit_cast<T>(name<S>(reinterpret_cast<InvokedT *>(addr),             \
                               bit_cast<InvokedT>(x), order, scope));          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN long long name(long long *addr, long long x,         \
                                         memory_order order,                   \
                                         memory_scope scope) noexcept {        \
    return bitcast_name<S, unsigned long long>(addr, x, order, scope);         \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN long name(long *addr, long x, memory_order order,    \
                                    memory_scope scope) noexcept {             \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, int>(addr, x, order, scope);                      \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN unsigned long name(                                  \
      unsigned long *addr, unsigned long x, memory_order order,                \
      memory_scope scope) noexcept {                                           \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, unsigned int>(addr, x, order, scope);             \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }

// Defines overloads for integral atomics for types that are not natively
// supported using bitcasts (long and unsigned long), for operations
// where the sign of the operands matters (e.g. atomicMin).
#define HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_SIGNED(     \
    name, bitcast_name)                                                        \
  template <access::address_space S, class InvokedT, class T>                  \
  HIPSYCL_HIPLIKE_BUILTIN T bitcast_name(T *addr, T x, memory_order order,     \
                                         memory_scope scope) noexcept {        \
    return bit_cast<T>(name<S>(reinterpret_cast<InvokedT *>(addr),             \
                               bit_cast<InvokedT>(x), order, scope));          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN long name(long *addr, long x, memory_order order,    \
                                    memory_scope scope) noexcept {             \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, int>(addr, x, order, scope);                      \
    } else {                                                                   \
      return bitcast_name<S, long long>(addr, x, order, scope);       \
    }                                                                          \
  }                                                                            \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN unsigned long name(                                  \
      unsigned long *addr, unsigned long x, memory_order order,                \
      memory_scope scope) noexcept {                                           \
    if constexpr (sizeof(long) == 4) {                                         \
      return bitcast_name<S, unsigned int>(addr, x, order, scope);             \
    } else {                                                                   \
      return bitcast_name<S, unsigned long long>(addr, x, order, scope);       \
    }                                                                          \
  }

// ******************** atomic fetch and ************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_and(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_and(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_and(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_AGNOSTIC(
    __acpp_atomic_fetch_and, __acpp_bitcast_atomic_fetch_and);

// ************************ atomic fetch or **********************


template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_or(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_or(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_or(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_AGNOSTIC(
    __acpp_atomic_fetch_or, __acpp_bitcast_atomic_fetch_or);

// ************************ atomic fetch xor **********************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_xor(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_xor(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_xor(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_AGNOSTIC(
    __acpp_atomic_fetch_xor, __acpp_bitcast_atomic_fetch_xor);

// ***************** Floating point and integral values *************


// *********************** atomic add ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_add(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_add(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_add(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

// atomicAdd supports float and double, so we only need to complete the integral
// overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_AGNOSTIC(
    __acpp_atomic_fetch_add, __acpp_bitcast_atomic_fetch_add);

// *********************** atomic sub ****************************

// Only supports int and unsigned
template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_sub(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_sub(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_sub(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

// Need CAS loop for 64bit types - the definition of this
// is required in order to be able to use the 
// HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS macro

#define HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(Type)                   \
  template <access::address_space S>                                           \
  HIPSYCL_HIPLIKE_BUILTIN Type __acpp_atomic_fetch_sub(                     \
      Type *addr, Type x, memory_order order, memory_scope scope) noexcept {   \
    Type old = __acpp_atomic_load<S>(addr, load_order_for_rmw(order), scope);\
    while (!__acpp_atomic_compare_exchange_strong<S>(                       \
        addr, old, old - x, order, failure_order_for_rmw(order), scope))     \
      ;                                                                        \
    return old;                                                                \
  }

HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(unsigned long long)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(float)
HIPSYCL_DEFINE_HIPLIKE_ATOMICSUB_CAS_EMULATION(double)

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN long
__acpp_atomic_fetch_sub(long *addr, long x, memory_order order,
                           memory_scope scope) noexcept {
  if constexpr (sizeof(long) == 4) {
    return static_cast<long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<int *>(addr), static_cast<int>(x), order, scope));
  } else {
    return static_cast<long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<long long *>(addr), static_cast<long long>(x), order,
        scope));
  }
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN unsigned long
__acpp_atomic_fetch_sub(unsigned long *addr, unsigned long x,
                           memory_order order, memory_scope scope) noexcept {
  if constexpr (sizeof(unsigned long) == 4) {
    return static_cast<unsigned long>(__acpp_atomic_fetch_sub<S>(
        reinterpret_cast<unsigned *>(addr), static_cast<unsigned>(x), order,
        scope));
  } else {
    return static_cast<unsigned long>(
        __acpp_atomic_fetch_sub<S>(
            reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(x), order, scope));
  }
}

// *********************** atomic min ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_min(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_min(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_min(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_SIGNED(
    __acpp_atomic_fetch_min, __acpp_bitcast_atomic_fetch_min);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_fetch_min(float *addr, float x, memory_order order,
                           memory_scope scope) noexcept {
  float old = __acpp_atomic_load<S>(addr, load_order_for_rmw(order), scope);
  do {
    if (old < x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, failure_order_for_rmw(order), scope));
  return x;
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_fetch_min(double *addr, double x, memory_order order,
                           memory_scope scope) noexcept {
  double old = __acpp_atomic_load<S>(addr, load_order_for_rmw(order), scope);
  do {
    if (old < x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, failure_order_for_rmw(order), scope));
  return x;
}

// *********************** atomic max ****************************

template <access::address_space S, class T>
HIPSYCL_HIPLIKE_BUILTIN T __acpp_atomic_fetch_max(
    T *addr, T x, memory_order order, memory_scope scope) noexcept {
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  return __scoped_atomic_fetch_max(
      addr, x, builtin_memory_order(order), builtin_memory_scope(scope));
#else
  __acpp_atomic_pre_fence(order, scope);
  const T result = __acpp_nvcxx_fetch_max(addr, x, scope);
  __acpp_atomic_post_fence(order, scope);
  return result;
#endif
}

// Need completion of integral overload set
HIPSYCL_DEFINE_HIPLIKE_ATOMIC_INTEGRAL_NONNATIVE_OVERLOADS_SIGNED(
    __acpp_atomic_fetch_max, __acpp_bitcast_atomic_fetch_max);

// Need CAS loop for emulation of floating point overloads
template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN float
__acpp_atomic_fetch_max(float *addr, float x, memory_order order,
                           memory_scope scope) noexcept {
  float old = __acpp_atomic_load<S>(addr, load_order_for_rmw(order), scope);
  do {
    if (old > x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, failure_order_for_rmw(order), scope));
  return x;
}

template <access::address_space S>
HIPSYCL_HIPLIKE_BUILTIN double
__acpp_atomic_fetch_max(double *addr, double x, memory_order order,
                           memory_scope scope) noexcept {
  double old = __acpp_atomic_load<S>(addr, load_order_for_rmw(order), scope);
  do {
    if (old > x)
      return old;
  } while (!__acpp_atomic_compare_exchange_strong<S>(
      addr, old, x, order, failure_order_for_rmw(order), scope));
  return x;
}

HIPSYCL_BUILTIN void
__acpp_atomic_fence(memory_order order, memory_scope scope) {
  if(order != memory_order::relaxed)
    __acpp_atomic_fence_impl(scope);
}

}
}
}


#endif

#endif
