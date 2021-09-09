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

#ifndef HIPSYCL_MULTI_PTR_HPP
#define HIPSYCL_MULTI_PTR_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
class accessor;

template <typename ElementType, access::address_space Space>
class multi_ptr
{
public:

  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;
  // Implementation defined pointer and reference types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions
  using pointer_t = ElementType*;
  using const_pointer_t = const ElementType*;
  using reference_t = ElementType&;
  using const_reference_t = const ElementType&;

  static constexpr access::address_space address_space = Space;
  // Constructors

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr()
    : _ptr{nullptr}
  {}

  multi_ptr(const multi_ptr&) = default;
  multi_ptr(multi_ptr&&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(ElementType* ptr)
    : _ptr{ptr}
  {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t)
    : _ptr{nullptr}
  {}

  // Assignment and access operators
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(const multi_ptr& other)
  {
    _ptr = other._ptr; 
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(multi_ptr&& other)
  {
    _ptr = other._ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(ElementType* ptr)
  {
    _ptr = ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(std::nullptr_t)
  {
    _ptr = nullptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  ElementType& operator*() const
  {
    return *_ptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  ElementType* operator->() const
  {
    return _ptr;
  }

  // Only if Space == global_space
  template <int dimensions,
            access::mode Mode,
            access::placeholder isPlaceholder,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::global_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::global_buffer, isPlaceholder> a)
    : _ptr{a.get_pointer()}
  {}

  // Only if Space == local_space
  template <int dimensions,
            access::mode Mode,
            access::placeholder isPlaceholder,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::local_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local, isPlaceholder> a)
    : _ptr{a.get_pointer()}
  {}

  // Only if Space == constant_space
  template <int dimensions,
            access::mode Mode,
            access::placeholder isPlaceholder,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::constant_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::constant_buffer, isPlaceholder> a)
    : _ptr{a.get_pointer()}
  {}

  // Returns the underlying OpenCL C pointer
  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get() const
  {
    return _ptr;
  }

  // Implicit conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET
  operator ElementType*() const
  {
    return _ptr;
  }

  // Explicit conversion to a multi_ptr<void>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<void, Space>() const
  {
    return multi_ptr<void, Space>{reinterpret_cast<void*>(_ptr)};
  }

  // Arithmetic operators
  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator++(multi_ptr<ElementType, Space>& mp)
  {
    ++(mp._ptr);
    return mp;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator++(multi_ptr<ElementType, Space>& mp, int)
  {
    multi_ptr old = mp;
    ++(mp._ptr);
    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator--(multi_ptr<ElementType, Space>& mp)
  {
    --(mp._ptr);
    return *mp;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator--(multi_ptr<ElementType, Space>& mp, int)
  {
    multi_ptr old = mp;
    --(mp._ptr);
    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator+=(multi_ptr<ElementType, Space>& lhs, difference_type r)
  {
    lhs._ptr += r;
    return lhs;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr& operator-=(multi_ptr<ElementType, Space>& lhs, difference_type r)
  {
    lhs._ptr -= r;
    return lhs;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator+(const multi_ptr<ElementType, Space>& lhs, difference_type r)
  {
    return multi_ptr{lhs._ptr + r};
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend multi_ptr operator-(const multi_ptr<ElementType, Space>& lhs, difference_type r)
  {
    return multi_ptr{lhs._ptr - r};
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const multi_ptr<ElementType, Space>& lhs,
                  const multi_ptr<ElementType, Space>& rhs)
  {
    return !(lhs == rhs);
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const multi_ptr<ElementType, Space>& lhs, std::nullptr_t)
  {
    return lhs.get() != nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(std::nullptr_t, const multi_ptr<ElementType, Space>& rhs)
  {
    return rhs != nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<(const multi_ptr<ElementType, Space>& lhs,
                const multi_ptr<ElementType, Space>& rhs)
  {
    return lhs.get() < rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<(const multi_ptr<ElementType, Space>& lhs, std::nullptr_t)
  {
    return lhs.get() < nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<(std::nullptr_t, const multi_ptr<ElementType, Space>& rhs)
  {
    return nullptr < rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>(const multi_ptr<ElementType, Space>& lhs,
                const multi_ptr<ElementType, Space>& rhs)
  {
    return lhs.get() > rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>(const multi_ptr<ElementType, Space>& lhs, std::nullptr_t)
  {
    return lhs.get() > nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>(std::nullptr_t, const multi_ptr<ElementType, Space>& rhs)
  {
    return nullptr < rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<=(const multi_ptr<ElementType, Space>& lhs,
                  const multi_ptr<ElementType, Space>& rhs)
  {
    return lhs.get() <= rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<=(const multi_ptr<ElementType, Space>& lhs, std::nullptr_t)
  {
    return lhs.get() <= nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<=(std::nullptr_t, const multi_ptr<ElementType, Space>& rhs)
  {
    return nullptr <= rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>=(const multi_ptr<ElementType, Space>& lhs,
                  const multi_ptr<ElementType, Space>& rhs)
  {
    return lhs.get() >= rhs.get();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>=(const multi_ptr<ElementType, Space>& lhs, std::nullptr_t)
  {
    return lhs.get() >= nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>=(std::nullptr_t, const multi_ptr<ElementType, Space>& rhs)
  {
    return nullptr >= rhs.get();
  }


  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const multi_ptr<ElementType, Space>& lhs,
                  const multi_ptr<ElementType, Space>& rhs)
  {
    return lhs.get() == rhs.get();
  }


  HIPSYCL_UNIVERSAL_TARGET
  void prefetch(size_t) const
  {}

private:
  ElementType* _ptr;
};

// Specialization of multi_ptr for void

template <access::address_space Space>
class multi_ptr<void, Space> {

public:

  using element_type = void;
  using difference_type = std::ptrdiff_t;
  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions

  using pointer_t = void*;
  using const_pointer_t = const void*;

  static constexpr access::address_space address_space = Space;
  // Constructors

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr()
    : _ptr{nullptr}
  {}

  multi_ptr(const multi_ptr& other) = default;
  multi_ptr(multi_ptr&& other) = default;

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(void* ptr)
    : _ptr{ptr}
  {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t)
    : _ptr{nullptr}
  {}

  // Assignment operators

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(const multi_ptr& other)
  { _ptr = other._ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(multi_ptr&& other)
  { _ptr = other._ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(void* ptr)
  { _ptr = ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(std::nullptr_t)
  { _ptr = nullptr; }


  // Only if Space == global_space
  template <typename ElementType,
            int dimensions,
            access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::global_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType,
                     dimensions,
                     Mode,
                     access::target::global_buffer,
                     access::placeholder::false_t> a)
    : _ptr{reinterpret_cast<void*>(a.get_pointer())}
  {}

  // Only if Space == local_space
  template <typename ElementType,
            int dimensions,
            access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::local_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType,
                     dimensions,
                     Mode,
                     access::target::local,
                     access::placeholder::false_t> a)
    : _ptr{reinterpret_cast<void*>(a.get_pointer())}
  {}

  // Only if Space == constant_space
  template <typename ElementType,
            int dimensions,
            access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S==access::address_space::constant_space>* = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType,
                     dimensions,
                     Mode,
                     access::target::constant_buffer,
                     access::placeholder::false_t> a)
    : _ptr{reinterpret_cast<void*>(a.get_pointer())}
  {}

  // Returns the underlying OpenCL C pointer
  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get() const
  {
    return _ptr;
  }
  // Implicit conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET
  operator void*() const
  {
    return _ptr;
  }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  HIPSYCL_UNIVERSAL_TARGET
  explicit operator multi_ptr<ElementType, Space>() const
  {
    return multi_ptr<ElementType, Space>{reinterpret_cast<ElementType*>(_ptr)};
  }
private:
  void* _ptr;
};

template <typename ElementType, access::address_space Space>
HIPSYCL_UNIVERSAL_TARGET
multi_ptr<ElementType, Space> make_ptr(ElementType* ptr)
{
  return multi_ptr<ElementType, Space>{ptr};
}

// Template specialization aliases for different pointer address spaces
template <typename ElementType>
using global_ptr = multi_ptr<ElementType, access::address_space::global_space>;

template <typename ElementType>
using local_ptr = multi_ptr<ElementType, access::address_space::local_space>;

template <typename ElementType>
using constant_ptr = multi_ptr<ElementType, access::address_space::constant_space>;

template <typename ElementType>
using private_ptr = multi_ptr<ElementType, access::address_space::private_space>;

// Deduction guides
template <int dimensions, access::mode Mode, access::placeholder isPlaceholder, class T>
multi_ptr( accessor<T, dimensions, Mode, access::target::global_buffer, isPlaceholder>)
-> multi_ptr<T, access::address_space::global_space>;

template <int dimensions, access::mode Mode, access::placeholder isPlaceholder, class T>
multi_ptr(accessor<T, dimensions, Mode, access::target::constant_buffer, isPlaceholder>)
-> multi_ptr<T, access::address_space::constant_space>;

template <int dimensions, access::mode Mode, access::placeholder isPlaceholder, class T>
multi_ptr(accessor<T, dimensions, Mode, access::target::local, isPlaceholder>)
-> multi_ptr<T, access::address_space::local_space>;

} // namespace sycl
} // namespace hipsycl

#endif
