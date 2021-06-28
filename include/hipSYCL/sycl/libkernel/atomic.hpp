/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_ATOMIC_HPP
#define HIPSYCL_ATOMIC_HPP

#include <type_traits>

#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/host/atomic_builtins.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"
#include "multi_ptr.hpp"


#include "atomic_builtins.hpp"

namespace hipsycl {
namespace sycl {


#ifdef HIPSYCL_EXT_FP_ATOMICS
  #define HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(template_param) \
    std::enable_if_t<std::is_integral<template_param>::value || std::is_floating_point<t>::value>* = nullptr
#else
  #define HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(template_param) \
    std::enable_if_t<std::is_integral<template_param>::value>* = nullptr
#endif


template <typename T, access::address_space addressSpace =
          access::address_space::global_space>
class atomic {
  static constexpr memory_scope default_scope() {
    if(addressSpace == access::address_space::global_space)
      return memory_scope::device;
    else if(addressSpace == access::address_space::local_space)
      return memory_scope::work_group;
    else if(addressSpace == access::address_space::private_space)
      return memory_scope::work_item;
    return memory_scope::device;
  }
public:
  template <typename pointerT>
  HIPSYCL_UNIVERSAL_TARGET
  atomic(multi_ptr<pointerT, addressSpace> ptr)
    : _ptr{reinterpret_cast<T*>(ptr.get())}
  {
    static_assert(sizeof(T) == sizeof(pointerT),
                  "Invalid pointer type for atomic<>");
  }

  HIPSYCL_KERNEL_TARGET
  void store(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile {
    detail::__hipsycl_atomic_store<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  HIPSYCL_KERNEL_TARGET
  T load(memory_order memoryOrder = memory_order::relaxed) const volatile {
    return detail::__hipsycl_atomic_load<addressSpace>(_ptr, memoryOrder,
                                                       default_scope());
  }

  HIPSYCL_KERNEL_TARGET
  T exchange(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_exchange<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  HIPSYCL_KERNEL_TARGET
  bool compare_exchange_strong(T &expected, T desired,
                               memory_order successMemoryOrder = memory_order::relaxed,
                               memory_order failMemoryOrder = memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_compare_exchange_strong<addressSpace>(
        _ptr, expected, desired, successMemoryOrder, failMemoryOrder,
        default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_add(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_add<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_sub(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_sub<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_and(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_and<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_or(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_or<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_xor(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_xor<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_min(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_min<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

  /* Available only when: T != float */
  template<class t = T,
           HIPSYCL_CONDITIONALLY_ENABLE_ATOMICS(t)>
  HIPSYCL_KERNEL_TARGET
  T fetch_max(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return detail::__hipsycl_atomic_fetch_max<addressSpace>(
        _ptr, operand, memoryOrder, default_scope());
  }

private:
  T* _ptr;
};


template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
void atomic_store(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  object.store(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_load(atomic<T, addressSpace> object, memory_order memoryOrder =
              memory_order::relaxed)
{
  return object.load(memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_exchange(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  return object.exchange(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
bool atomic_compare_exchange_strong(atomic<T, addressSpace> object, T &expected, T desired,
                                    memory_order successMemoryOrder = memory_order::relaxed,
                                    memory_order failMemoryOrder = memory_order::relaxed)
{
  return object.compare_exchange_strong(expected, desired,
                                        successMemoryOrder, failMemoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_add(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_add(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_sub(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_sub(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_and(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_and(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_or(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  return object.fetch_or(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_xor(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_xor(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_min(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_min(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
HIPSYCL_KERNEL_TARGET
T atomic_fetch_max(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_max(operand, memoryOrder);
}

} // namespace sycl
} // namespace hipsycl

#endif
