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

#ifndef HIPSYCL_ATOMIC_BUILTINS_HPP
#define HIPSYCL_ATOMIC_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/builtin_dispatch.hpp"

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/atomic_builtins.hpp"
#endif
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                 \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "generic/hiplike/atomic_builtins.hpp"
#endif
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
#include "spirv/atomic_builtins.hpp"
#endif



namespace hipsycl {
namespace sycl {
namespace detail {

template <access::address_space S, class T>
HIPSYCL_BUILTIN void __hipsycl_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  HIPSYCL_DISPATCH_BUILTIN(__hipsycl_atomic_store<S>, addr, x, order, scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_load<S>, addr, order, scope);
}


template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_exchange<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_compare_exchange_weak<S>,
                                  addr, expected, desired, success, failure,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __hipsycl_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_compare_exchange_strong<S>,
                                  addr, expected, desired, success, failure,
                                  scope);
}

// Integral values only

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_and<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_or<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_xor<S>, addr, x, order,
                                  scope);
}

// Floating point and integral values

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_add<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_sub<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_min<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __hipsycl_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__hipsycl_atomic_fetch_max<S>, addr, x, order,
                                  scope);
}


}
}
}

#endif
