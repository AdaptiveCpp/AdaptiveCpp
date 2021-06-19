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


template <class T, access::address_space S>
HIPSYCL_BUILTIN void __hipsycl_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  __atomic_store_n(addr, x, builtin_memory_order(order));
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {
  return __atomic_load_n(addr, builtin_memory_order(order));
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  return atomicExch(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  T old = expected;
  expected = atomicCAS(addr, expected, desired);
  return old == expected;

}

template <class T, access::address_space S>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {

  T old = expected;
  expected = atomicCAS(addr, expected, desired);
  return old == expected;

}

// Integral values only

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicAnd(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicOr(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicXor(addr, x);
}

// Floating point and integral values

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicAdd(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   return atomicSub(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicMin(addr, x);
}

template <class T, access::address_space S>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  return atomicMax(addr, x);
}

}
}
}

#endif

#endif
