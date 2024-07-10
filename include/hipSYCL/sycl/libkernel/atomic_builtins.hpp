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
#ifndef HIPSYCL_ATOMIC_BUILTINS_HPP
#define HIPSYCL_ATOMIC_BUILTINS_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/builtin_dispatch.hpp"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
#include "host/atomic_builtins.hpp"
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA ||                                 \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
#include "generic/hiplike/atomic_builtins.hpp"
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/atomic_builtins.hpp"
#endif



namespace hipsycl {
namespace sycl {
namespace detail {

template <access::address_space S, class T>
HIPSYCL_BUILTIN void __acpp_atomic_store(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  HIPSYCL_DISPATCH_BUILTIN(__acpp_atomic_store<S>, addr, x, order, scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_load(T *addr, memory_order order,
                                        memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_load<S>, addr, order, scope);
}


template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_exchange(T *addr, T x, memory_order order,
                                            memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_exchange<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_weak(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_compare_exchange_weak<S>,
                                  addr, expected, desired, success, failure,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN bool __acpp_atomic_compare_exchange_strong(
    T *addr, T &expected, T desired, memory_order success, memory_order failure,
    memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_compare_exchange_strong<S>,
                                  addr, expected, desired, success, failure,
                                  scope);
}

// Integral values only

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_and(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_and<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_or(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_or<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_xor(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_xor<S>, addr, x, order,
                                  scope);
}

// Floating point and integral values

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_add(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_add<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_sub(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
   HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_sub<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_min(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_min<S>, addr, x, order,
                                  scope);
}

template <access::address_space S, class T>
HIPSYCL_BUILTIN T __acpp_atomic_fetch_max(T *addr, T x, memory_order order,
                                             memory_scope scope) noexcept {
  HIPSYCL_RETURN_DISPATCH_BUILTIN(__acpp_atomic_fetch_max<S>, addr, x, order,
                                  scope);
}


}
}
}

#endif
