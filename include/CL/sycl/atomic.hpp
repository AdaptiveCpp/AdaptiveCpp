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

#include <type_traits>

#include "access.hpp"
#include "multi_ptr.hpp"
#include "backend/backend.hpp"


namespace cl {
namespace sycl {



enum class memory_order : int
{
  relaxed
};

template <typename T, access::address_space addressSpace =
          access::address_space::global_space>
class atomic {
public:
  template <typename pointerT>
  __host__ __device__
  atomic(multi_ptr<pointerT, addressSpace> ptr)
    : _ptr{reinterpret_cast<T*>(ptr.get())}
  {
    static_assert(sizeof(T) == sizeof(pointerT),
                  "Invalid pointer type for atomic<>");
  }

  /// \todo unimplemented
  __device__
  void store(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile;

  /// \todo unimplemented
  __device__
  T load(memory_order memoryOrder = memory_order::relaxed) const volatile;

  __device__
  T exchange(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicExch(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  /// \todo unimplemented
  __device__
  bool compare_exchange_strong(T &expected, T desired,
                               memory_order successMemoryOrder = memory_order::relaxed,
                               memory_order failMemoryOrder = memory_order::relaxed) volatile;

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_add(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicAdd(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_sub(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicSub(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_and(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicAnd(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_or(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicOr(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_xor(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicXor(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  t fetch_min(t operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicMin(_ptr.get(), operand);
  }

  /* Available only when: T != float */
  template<class t = T,
           std::enable_if_t<!std::is_floating_point<t>::value>* = nullptr>
  __device__
  T fetch_max(T operand, memory_order memoryOrder =
      memory_order::relaxed) volatile
  {
    return atomicMax(_ptr.get(), operand);
  }

private:
  multi_ptr<T, addressSpace> _ptr;
};


template <typename T, access::address_space addressSpace>
__device__
void atomic_store(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  object.store(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_load(atomic<T, addressSpace> object, memory_order memoryOrder =
              memory_order::relaxed)
{
  return object.load(memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_exchange(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  return object.exchange(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
bool atomic_compare_exchange_strong(atomic<T, addressSpace> object, T &expected, T desired,
                                    memory_order successMemoryOrder = memory_order::relaxed,
                                    memory_order failMemoryOrder = memory_order::relaxed)
{
  return object.compare_exchange_strong(expected, desired,
                                        successMemoryOrder, failMemoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_add(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_add(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_sub(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_sub(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_and(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_and(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_or(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                  memory_order::relaxed)
{
  return object.fetch_or(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_xor(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_xor(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_min(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_min(operand, memoryOrder);
}

template <typename T, access::address_space addressSpace>
__device__
T atomic_fetch_max(atomic<T, addressSpace> object, T operand, memory_order memoryOrder =
                   memory_order::relaxed)
{
  return object.fetch_max(operand, memoryOrder);
}

} // namespace sycl
} // namespace cl
