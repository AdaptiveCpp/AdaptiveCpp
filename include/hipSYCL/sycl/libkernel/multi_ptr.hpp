/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_MULTI_PTR_HPP
#define HIPSYCL_MULTI_PTR_HPP

#include "hipSYCL/sycl/access.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "hipSYCL/sycl/libkernel/memory.hpp"

#include <cstddef>
#include <type_traits>

#define ACPP_MULTIPTR_ARITHMETIC_OPS                                           \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr &operator++(multi_ptr &mp) {                                \
    ++(mp._ptr);                                                               \
    return mp;                                                                 \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr operator++(multi_ptr &mp, int) {                            \
    multi_ptr old = mp;                                                        \
    ++(mp._ptr);                                                               \
    return old;                                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr &operator--(multi_ptr &mp) {                                \
    --(mp._ptr);                                                               \
    return mp;                                                                 \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr operator--(multi_ptr &mp, int) {                            \
    multi_ptr old = mp;                                                        \
    --(mp._ptr);                                                               \
    return old;                                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr &operator+=(multi_ptr &lhs, difference_type r) {            \
    lhs._ptr += r;                                                             \
    return lhs;                                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr &operator-=(multi_ptr &lhs, difference_type r) {            \
    lhs._ptr -= r;                                                             \
    return lhs;                                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr operator+(const multi_ptr &lhs, difference_type r) {        \
    return multi_ptr{lhs._ptr + r};                                            \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend multi_ptr operator-(const multi_ptr &lhs, difference_type r) {        \
    return multi_ptr{lhs._ptr - r};                                            \
  }

#define ACPP_MULTIPTR_NULLPTR_COMP                                             \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator==(std::nullptr_t,              \
                                                  multi_ptr rhs) {             \
    return rhs.get() == nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator==(multi_ptr lhs,               \
                                                  std::nullptr_t) {            \
    return lhs.get() == nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator!=(std::nullptr_t,              \
                                                  multi_ptr rhs) {             \
    return rhs.get() != nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator!=(multi_ptr lhs,               \
                                                  std::nullptr_t) {            \
    return lhs.get() != nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator<(std::nullptr_t,               \
                                                 multi_ptr rhs) {              \
    return rhs.get() != nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator<(multi_ptr lhs,                \
                                                 std::nullptr_t) {             \
    return false;                                                              \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator>(std::nullptr_t,               \
                                                 multi_ptr rhs) {              \
    return false;                                                              \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator>(multi_ptr lhs,                \
                                                 std::nullptr_t) {             \
    return lhs.get() != nullptr;                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator<=(std::nullptr_t,              \
                                                  multi_ptr rhs) {             \
    return true;                                                               \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator<=(multi_ptr lhs,               \
                                                  std::nullptr_t) {            \
    return lhs._ptr == nullptr;                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator>=(std::nullptr_t,              \
                                                  multi_ptr rhs) {             \
    return rhs._ptr == nullptr;                                                \
  }                                                                            \
                                                                               \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator>=(multi_ptr lhs,               \
                                                  std::nullptr_t) {            \
    return true;                                                               \
  }

#define ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(op)                              \
  HIPSYCL_UNIVERSAL_TARGET friend bool operator op(multi_ptr lhs,              \
                                                   multi_ptr rhs) {            \
    return lhs._ptr op rhs._ptr;                                               \
  }

#define ACPP_MULTIPTR_MULTIPTR_COMP                                            \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(==)                                    \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(!=)                                    \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(<)                                     \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(>)                                     \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(<=)                                    \
  ACPP_DEFINE_COMP_OP_MULTIPTR_MULTIPTR(>=)

namespace hipsycl {
namespace sycl {

template <typename dataT, int dimensions, access::mode accessmode,
          access::target accessTarget, access::placeholder isPlaceholder>
class accessor;

template <typename dataT, int dimensions = 1>
using local_accessor =
    accessor<dataT, dimensions, access::mode::read_write, access::target::local,
             access::placeholder::false_t>;

template <typename T> struct remove_decoration {
  using type = T;
};

template <typename T>
using remove_decoration_t = typename remove_decoration<T>::type;

namespace detail {
template <access::decorated>
inline constexpr access::decorated NegateDecorated = access::decorated::legacy;

template <>
inline constexpr access::decorated NegateDecorated<access::decorated::no> =
    access::decorated::yes;

template <>
inline constexpr access::decorated NegateDecorated<access::decorated::yes> =
    access::decorated::no;
} // namespace detail

template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
class multi_ptr {
  template <typename, access::address_space, access::decorated>
  friend class multi_ptr;

public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = ElementType;
  using pointer = std::add_pointer_t<value_type>;
  using reference = std::add_lvalue_reference_t<value_type>;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  static_assert(std::is_same_v<remove_decoration_t<pointer>,
                               std::add_pointer_t<value_type>>);
  static_assert(std::is_same_v<remove_decoration_t<reference>,
                               std::add_lvalue_reference_t<value_type>>);
  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(typename multi_ptr<ElementType, Space,
                                        access::decorated::yes>::pointer ptr)
      : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Available only when:
  //   (Space == access::address_space::global_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_same_v<std::remove_const_t<ElementType>,
  //   std::remove_const_t<AccDataT>>) && (std::is_const_v<ElementType> ||
  //    !std::is_const_v<accessor<AccDataT, Dimensions, Mode, target::device,
  //                              IsPlaceholder>::value_type>)
  template <
      typename AccDataT, int Dimensions, access_mode Mode,
      access::placeholder IsPlaceholder, access::address_space S = Space,
      typename E = ElementType,
      std::enable_if_t<(S == access::address_space::global_space ||
                        S == access::address_space::generic_space) &&
                           std::is_same_v<std::remove_const_t<E>,
                                          std::remove_const_t<AccDataT>> &&
                           (std::is_const_v<E> ||
                            !std::is_const_v<typename accessor<
                                AccDataT, Dimensions, Mode, target::device,
                                IsPlaceholder>::value_type>),
                       bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::device, IsPlaceholder> a)
      : _ptr{a.get_pointer()} {}

  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_same_v<std::remove_const_t<ElementType>,
  //   std::remove_const_t<AccDataT>>) && (std::is_const_v<ElementType> ||
  //   !std::is_const_v<AccDataT>)
  template <
      typename AccDataT, int Dimensions, access::address_space S = Space,
      typename E = ElementType,
      std::enable_if_t<(S == access::address_space::local_space ||
                        S == access::address_space::generic_space) &&
                           std::is_same_v<std::remove_const_t<E>,
                                          std::remove_const_t<AccDataT>> &&
                           (std::is_const_v<E> || !std::is_const_v<AccDataT>),
                       bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(local_accessor<AccDataT, Dimensions> a)
      : _ptr{a.get_pointer()} {}

  // Deprecated
  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_same_v<std::remove_const_t<ElementType>,
  //   std::remove_const_t<AccDataT>>) && (std::is_const_v<ElementType> ||
  //   !std::is_const_v<AccDataT>)
  template <
      typename AccDataT, int Dimensions, access_mode Mode,
      access::placeholder IsPlaceholder, access::address_space S = Space,
      typename E = ElementType,
      std::enable_if_t<(S == access::address_space::local_space ||
                        S == access::address_space::generic_space) &&
                           std::is_same_v<std::remove_const_t<E>,
                                          std::remove_const_t<AccDataT>> &&
                           (std::is_const_v<E> || !std::is_const_v<AccDataT>),
                       bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<AccDataT, Dimensions, Mode, target::local, IsPlaceholder> a)
      : _ptr{a.get_pointer()} {}

  // Deprecated
  // Available only when:
  //   Space == access::address_space::constant_space &&
  //   (std::is_same_v<std::remove_const_t<ElementType>,
  //   std::remove_const_t<AccDataT>>) && (std::is_const_v<ElementType> ||
  //   !std::is_const_v<AccDataT>)
  template <
      typename AccDataT, int Dimensions, access::placeholder IsPlaceholder,
      access::address_space S = Space, typename E = ElementType,
      std::enable_if_t<S == access::address_space::constant_space &&
                           std::is_same_v<std::remove_const_t<E>,
                                          std::remove_const_t<AccDataT>> &&
                           (std::is_const_v<E> || !std::is_const_v<AccDataT>),
                       bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<AccDataT, Dimensions, access_mode::read,
                     target::constant_buffer, IsPlaceholder>
                a)
      : _ptr{a.get_pointer()} {}

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  // Available only when:
  //   (Space == access::address_space::generic_space &&
  //    AS != access::address_space::constant_space)
  template <access::address_space AS, access::decorated IsDecorated,
            access::address_space S = Space,
            std::enable_if_t<S == access::address_space::generic_space &&
                                 AS != access::address_space::constant_space,
                             bool> = true>
  multi_ptr &operator=(const multi_ptr<value_type, AS, IsDecorated> &other) {
    _ptr = other._ptr;
    return *this;
  }

  // Available only when:
  //   (Space == access::address_space::generic_space &&
  //    AS != access::address_space::constant_space)
  template <access::address_space AS, access::decorated IsDecorated,
            access::address_space S = Space,
            std::enable_if_t<S == access::address_space::generic_space &&
                                 AS != access::address_space::constant_space,
                             bool> = true>
  multi_ptr &operator=(multi_ptr<value_type, AS, IsDecorated> &&other) {
    _ptr = other._ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET reference operator[](std::ptrdiff_t index) const {
    return _ptr[index];
  }

  HIPSYCL_UNIVERSAL_TARGET reference operator*() const { return *_ptr; }
  HIPSYCL_UNIVERSAL_TARGET pointer operator->() const { return _ptr; }
  HIPSYCL_UNIVERSAL_TARGET pointer get() const { return _ptr; }
  HIPSYCL_UNIVERSAL_TARGET pointer get_raw() const { return _ptr; }
  HIPSYCL_UNIVERSAL_TARGET pointer get_decorated() const { return _ptr; }

  // Conversion to the underlying pointer type
  // Deprecated, get() should be used instead.
  [[deprecated("Use get() instead")]] HIPSYCL_UNIVERSAL_TARGET
  operator pointer() const {
    return get();
  }

  // Cast to private_ptr
  // Available only when: (Space == access::address_space::generic_space)
  template <access::decorated IsDecorated, access::address_space S = Space,
            typename V = value_type,
            std::enable_if_t<S == access::address_space::generic_space &&
                                 !std::is_const_v<V>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<value_type, access::address_space::private_space,
                     IsDecorated>() const {
    return multi_ptr<value_type, access::address_space::private_space,
                     IsDecorated>{_ptr};
  }

  // Cast to private_ptr of const data
  // Available only when: (Space == access::address_space::generic_space)
  template <
      access::decorated IsDecorated, access::address_space S = Space,
      std::enable_if_t<S == access::address_space::generic_space, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<const value_type, access::address_space::private_space,
                     IsDecorated>() const {
    return multi_ptr<const value_type, access::address_space::private_space,
                     IsDecorated>{_ptr};
  }

  // Cast to global_ptr
  // Available only when: (Space == access::address_space::generic_space)
  template <
      access::decorated IsDecorated, access::address_space S = Space,
      std::enable_if_t<S == access::address_space::generic_space, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<value_type, access::address_space::global_space,
                     IsDecorated>() const {
    return multi_ptr<value_type, access::address_space::global_space,
                     IsDecorated>{_ptr};
  }

  // Cast to global_ptr of const data
  // Available only when: (Space == access::address_space::generic_space)
  template <access::decorated IsDecorated, access::address_space S = Space,
            typename V = value_type,
            std::enable_if_t<S == access::address_space::generic_space &&
                                 !std::is_const_v<V>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<const value_type, access::address_space::global_space,
                     IsDecorated>() const {
    return multi_ptr<const value_type, access::address_space::global_space,
                     IsDecorated>{_ptr};
  }

  // Cast to local_ptr
  // Available only when: (Space == access::address_space::generic_space)
  template <
      access::decorated IsDecorated, access::address_space S = Space,
      std::enable_if_t<S == access::address_space::generic_space, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<value_type, access::address_space::local_space,
                     IsDecorated>() const {
    return multi_ptr<value_type, access::address_space::local_space,
                     IsDecorated>{_ptr};
  }

  // Cast to local_ptr of const data
  // Available only when: (Space == access::address_space::generic_space)
  template <access::decorated IsDecorated, access::address_space S = Space,
            typename V = value_type,
            std::enable_if_t<S == access::address_space::generic_space &&
                                 !std::is_const_v<V>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<const value_type, access::address_space::local_space,
                     IsDecorated>() const {
    return multi_ptr<const value_type, access::address_space::local_space,
                     IsDecorated>{_ptr};
  }

  // Implicit conversion to a multi_ptr<void>.
  // Available only when: (!std::is_const_v<value_type>)
  template <access::decorated IsDecorated, typename V = value_type,
            std::enable_if_t<!std::is_const_v<V>, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<void, Space, IsDecorated>() const {
    return multi_ptr<void, Space, IsDecorated>{reinterpret_cast<void *>(_ptr)};
  }

  // Implicit conversion to a multi_ptr<const void>.
  // Available only when: (std::is_const_v<value_type>)
  template <access::decorated IsDecorated, typename V = value_type,
            std::enable_if_t<std::is_const_v<V>, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const void, Space, IsDecorated>() const {
    return multi_ptr<const void, Space, IsDecorated>{
        reinterpret_cast<const void *>(_ptr)};
  }

  // Implicit conversion to multi_ptr<const value_type, Space>.
  template <access::decorated IsDecorated>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const value_type, Space, IsDecorated>() const {
    return multi_ptr<const value_type, Space, IsDecorated>{_ptr};
  }

  // Implicit conversion to the (non-)decorated version of multi_ptr.
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>{_ptr};
  }

  // Available only when: (Space == address_space::global_space)
  template <
      access::address_space S = Space,
      std::enable_if_t<S == access::address_space::global_space, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET void prefetch(size_t numElements) const {}

  ACPP_MULTIPTR_ARITHMETIC_OPS
  ACPP_MULTIPTR_NULLPTR_COMP
  ACPP_MULTIPTR_MULTIPTR_COMP

private:
  ElementType *_ptr;
};

// Specialization of multi_ptr for void
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<void, Space, DecorateAddress> {
public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = void;
  using pointer = void *;
  using difference_type = std::ptrdiff_t;

  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(void *ptr) : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Available only when:
  //   (Space == access::address_space::global_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> ||
  //    !std::is_const_v<accessor<ElementType, Dimensions, Mode, target::device,
  //                              IsPlaceholder>::value_type>)
  template <
      typename ElementType, int Dimensions, access_mode Mode,
      access::placeholder IsPlaceholder, access::address_space S = Space,
      std::enable_if_t<(S == access::address_space::global_space ||
                        S == access::address_space::generic_space) &&
                           !std::is_const_v<typename accessor<
                               ElementType, Dimensions, Mode, target::device,
                               IsPlaceholder>::value_type>,
                       bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> a)
      : _ptr{reinterpret_cast<void *>(
            a.template get_multi_ptr<DecorateAddress>().get())} {}

  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <typename ElementType, int Dimensions,
            access::address_space S = Space,
            std::enable_if_t<(S == access::address_space::local_space ||
                              S == access::address_space::generic_space) &&
                                 !std::is_const_v<ElementType>,
                             bool> = true>
  multi_ptr(local_accessor<ElementType, Dimensions> a)
      : _ptr{reinterpret_cast<void *>(
            a.template get_multi_ptr<DecorateAddress>().get())} {}

  // Deprecated
  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder, access::address_space S = Space,
            std::enable_if_t<(S == access::address_space::local_space ||
                              S == access::address_space::generic_space) &&
                                 !std::is_const_v<ElementType>,
                             bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> a)
      : _ptr{reinterpret_cast<void *>(a.get_pointer().get())} {}

  // Deprecated
  // Available only when:
  //   Space == access::address_space::constant_space &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <typename ElementType, int Dimensions,
            access::placeholder IsPlaceholder, access::address_space S = Space,
            std::enable_if_t<(S == access::address_space::constant_space) &&
                                 !std::is_const_v<ElementType>,
                             bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, Dimensions, access_mode::read,
                     target::constant_buffer, IsPlaceholder>
                a)
      : _ptr{reinterpret_cast<void *>(a.get_pointer().get())} {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  HIPSYCL_UNIVERSAL_TARGET multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET pointer get() const { return _ptr; }

  // Conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET operator pointer() const { return get(); }

  // Explicit conversion to a multi_ptr<ElementType>
  // Available only when: (std::is_const_v<ElementType> ||
  // !std::is_const_v<void>)
  template <typename ElementType>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    return multi_ptr<ElementType, Space, DecorateAddress>{_ptr};
  }

  // Implicit conversion to the non-decorated version of multi_ptr.
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>{_ptr};
  }

  // Implicit conversion to multi_ptr<const void, Space>
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const void, Space, DecorateAddress>() const {
    return multi_ptr<const void, Space, DecorateAddress>{_ptr};
  }

  ACPP_MULTIPTR_MULTIPTR_COMP
  ACPP_MULTIPTR_NULLPTR_COMP

private:
  void *_ptr;
};

// Specialization of multi_ptr for const void
template <access::address_space Space, access::decorated DecorateAddress>
class multi_ptr<const void, Space, DecorateAddress> {
public:
  static constexpr bool is_decorated =
      DecorateAddress == access::decorated::yes;
  static constexpr access::address_space address_space = Space;

  using value_type = const void;
  using pointer = const void *;
  using difference_type = std::ptrdiff_t;

  // Legacy has a different interface.
  static_assert(DecorateAddress != access::decorated::legacy);

  // Constructors
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(const void *ptr) : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Available only when:
  //   (Space == access::address_space::global_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> ||
  //    !std::is_const_v<accessor<ElementType, Dimensions, Mode, target::device,
  //                              IsPlaceholder>::value_type>)
  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder, access::address_space S = Space,
            std::enable_if_t<S == access::address_space::global_space ||
                                 S == access::address_space::generic_space,
                             bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> a)
      : _ptr{reinterpret_cast<const void *>(
            a.template get_multi_pointer<DecorateAddress>().get())} {}

  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <typename ElementType, int Dimensions,
            access::address_space S = Space,
            std::enable_if_t<S == access::address_space::local_space ||
                                 S == access::address_space::generic_space,
                             bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(local_accessor<ElementType, Dimensions> a)
      : _ptr{reinterpret_cast<const void *>(
            a.template get_multi_pointer<DecorateAddress>().get())} {}

  // Deprecated
  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space) &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <typename ElementType, int Dimensions, access_mode Mode,
            access::placeholder IsPlaceholder, access::address_space S = Space,
            std::enable_if_t<S == access::address_space::local_space ||
                                 S == access::address_space::generic_space,
                             bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> a)
      : _ptr{reinterpret_cast<const void *>(a.get_pointer().get())} {}

  // Deprecated
  // Available only when:
  //   Space == access::address_space::constant_space &&
  //   (std::is_const_v<VoidType> || !std::is_const_v<ElementType>)
  template <
      typename ElementType, int Dimensions, access::placeholder IsPlaceholder,
      access::address_space S = Space,
      std::enable_if_t<S == access::address_space::constant_space, bool> = true>
  [[deprecated]] HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, Dimensions, access_mode::read,
                     target::constant_buffer, IsPlaceholder>
                a)
      : _ptr{reinterpret_cast<const void *>(a.get_pointer().get())} {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;
  HIPSYCL_UNIVERSAL_TARGET multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET pointer get() const { return _ptr; }

  // Conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET operator pointer() const { return get(); }

  // Explicit conversion to a multi_ptr<ElementType>
  // Available only when: (std::is_const_v<ElementType> ||
  // !std::is_const_v<VoidType>)
  template <typename ElementType,
            std::enable_if_t<std::is_const_v<ElementType>, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<ElementType, Space, DecorateAddress>() const {
    return multi_ptr<ElementType, Space, DecorateAddress>{_ptr};
  }

  // Implicit conversion to the non-decorated version of multi_ptr.
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>() const {
    return multi_ptr<value_type, Space,
                     detail::NegateDecorated<DecorateAddress>>{_ptr};
  }

  ACPP_MULTIPTR_MULTIPTR_COMP
  ACPP_MULTIPTR_NULLPTR_COMP

private:
  const void *_ptr;
};

// Legacy interfaces
template <typename ElementType, access::address_space Space>
class [[deprecated]] multi_ptr<ElementType, Space, access::decorated::legacy> {
public:
  using value_type = ElementType;
  using element_type = ElementType;
  using difference_type = std::ptrdiff_t;

  using pointer_t =
      typename multi_ptr<ElementType, Space, access::decorated::yes>::pointer;
  using const_pointer_t = typename multi_ptr<const ElementType, Space,
                                             access::decorated::yes>::pointer;
  using reference_t =
      typename multi_ptr<ElementType, Space, access::decorated::yes>::reference;
  using const_reference_t =
      typename multi_ptr<const ElementType, Space,
                         access::decorated::yes>::reference;

  static constexpr access::address_space address_space = Space;

  // Constructors
  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &) = default;
  multi_ptr(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(ElementType *ptr) : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Assignment and access operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(ElementType *ptr) {
    _ptr = ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend ElementType &operator*(const multi_ptr &mp) { return *mp._ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  ElementType *operator->() const { return _ptr; }

  // Available only when:
  //   (Space == access::address_space::global_space ||
  //    Space == access::address_space::generic_space)
  template <int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
            access::address_space S = Space,
            std::enable_if_t<S == access::address_space::global_space ||
                                 S == access::address_space::generic_space,
                             bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::device, IsPlaceholder> a)
      : _ptr{a.get_pointer()} {}

  // Available only when:
  //   (Space == access::address_space::local_space ||
  //    Space == access::address_space::generic_space)
  template <
      int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
      access::address_space S = Space,
      std::enable_if_t<(S == access::address_space::local_space ||
                        S == access::address_space::generic_space) &&
                           !(Mode == access::mode::read_write &&
                             IsPlaceholder == access::placeholder::false_t),
                       bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, Dimensions, Mode, target::local, IsPlaceholder> a)
      : _ptr{a.get_pointer()} {}

  // // Available only when:
  // //   (Space == access::address_space::local_space ||
  // //    Space == access::address_space::generic_space) &&
  // //   (std::is_same_v<std::remove_const_t<ElementType>,
  // //   std::remove_const_t<AccDataT>>) && (std::is_const_v<ElementType> ||
  // //   !std::is_const_v<AccDataT>)
  template <
      typename AccDataT, int Dimensions, access::address_space S = Space,
      typename E = ElementType,
      std::enable_if_t<(S == access::address_space::local_space ||
                        S == access::address_space::generic_space) &&
                           std::is_same_v<std::remove_const_t<E>,
                                          std::remove_const_t<AccDataT>> &&
                           (std::is_const_v<E> || !std::is_const_v<AccDataT>),
                       bool> = true>
  HIPSYCL_KERNEL_TARGET multi_ptr(local_accessor<AccDataT, Dimensions> a)
      : _ptr{a.get_pointer()} {}

  // Only if Space == constant_space
  template <
      int Dimensions, access_mode Mode, access::placeholder IsPlaceholder,
      access::address_space S = Space,
      std::enable_if_t<S == access::address_space::constant_space, bool> = true>
  multi_ptr(accessor<ElementType, Dimensions, Mode, target::constant_buffer,
                     IsPlaceholder>
                a)
      : _ptr{a.get_pointer()} {}

  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get() const { return _ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  std::add_pointer_t<value_type> get_raw() const { return _ptr; }

  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get_decorated() const { return _ptr; }

  // Implicit conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET
  operator ElementType *() const { return _ptr; }

  // Implicit conversion to a multi_ptr<void>
  // Available only when ElementType is not const-qualified
  HIPSYCL_UNIVERSAL_TARGET
  template <typename E = ElementType,
            std::enable_if_t<!std::is_const_v<E>, bool> = true>
  operator multi_ptr<void, Space, access::decorated::legacy>() const {
    return multi_ptr<void, Space, access::decorated::legacy>{
        reinterpret_cast<void *>(_ptr)};
  }

  // Implicit conversion to a multi_ptr<const void>
  // Available only when ElementType is const-qualified
  HIPSYCL_UNIVERSAL_TARGET
  template <typename E = ElementType,
            std::enable_if_t<std::is_const_v<E>, bool> = true>
  operator multi_ptr<const void, Space, access::decorated::legacy>() const {
    return multi_ptr<const void, Space, access::decorated::legacy>{
        reinterpret_cast<const void *>(_ptr)};
  }

  // Implicit conversion to multi_ptr<const value_type, Space>.
  HIPSYCL_UNIVERSAL_TARGET
  operator multi_ptr<const ElementType, Space, access::decorated::legacy>()
      const {
    return multi_ptr<const ElementType, Space, access::decorated::legacy>{_ptr};
  }

  ACPP_MULTIPTR_ARITHMETIC_OPS
  ACPP_MULTIPTR_NULLPTR_COMP
  ACPP_MULTIPTR_MULTIPTR_COMP

  HIPSYCL_UNIVERSAL_TARGET
  void prefetch(size_t) const {}

private:
  ElementType *_ptr;
};

// Specialization of legacy multi_ptr for const void
template <access::address_space Space>
class [[deprecated]] multi_ptr<const void, Space, access::decorated::legacy> {
public:
  using element_type = const void;
  using difference_type = std::ptrdiff_t;
  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions

  using pointer_t = const void *;
  using const_pointer_t = const void *;

  static constexpr access::address_space address_space = Space;
  // Constructors

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &other) = default;
  multi_ptr(multi_ptr &&other) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(const void *ptr) : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Assignment operators

  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(void *ptr) {
    _ptr = ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  // Only if Space == global_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S == access::address_space::global_space>
                * = nullptr>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer,
               access::placeholder::false_t>
          a)
      : _ptr{reinterpret_cast<const void *>(a.get_pointer().get())} {}

  // Only if Space == local_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S == access::address_space::local_space>
                * = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     access::placeholder::false_t>
                a)
      : _ptr{reinterpret_cast<const void *>(a.get_pointer().get())} {}

  // Only if Space == constant_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<
                S == access::address_space::constant_space> * = nullptr>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer,
               access::placeholder::false_t>
          a)
      : _ptr{reinterpret_cast<const void *>(a.get_pointer().get())} {}

  // Returns the underlying OpenCL C pointer
  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get() const { return _ptr; }
  // Implicit conversion to the underlying pointer type
  HIPSYCL_UNIVERSAL_TARGET
  operator const void *() const { return _ptr; }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<ElementType, Space>() const {
    return multi_ptr<ElementType, Space>{reinterpret_cast<ElementType *>(_ptr)};
  }

  ACPP_MULTIPTR_MULTIPTR_COMP
  ACPP_MULTIPTR_NULLPTR_COMP

private:
  const void *_ptr;
};

// Specialization of legacy multi_ptr for void
template <access::address_space Space>
class [[deprecated]] multi_ptr<void, Space, access::decorated::legacy> {
public:
  using element_type = void;
  using difference_type = std::ptrdiff_t;
  // Implementation defined pointer types that correspond to
  // SYCL/OpenCL interoperability types for OpenCL C functions

  using pointer_t = void *;
  using const_pointer_t = const void *;

  static constexpr access::address_space address_space = Space;
  // Constructors

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr() : _ptr{nullptr} {}

  multi_ptr(const multi_ptr &other) = default;
  multi_ptr(multi_ptr &&other) = default;

  HIPSYCL_UNIVERSAL_TARGET
  explicit multi_ptr(void *ptr) : _ptr{ptr} {}

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr(std::nullptr_t) : _ptr{nullptr} {}

  // Assignment operators
  multi_ptr &operator=(const multi_ptr &) = default;
  multi_ptr &operator=(multi_ptr &&) = default;

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(void *ptr) {
    _ptr = ptr;
    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  multi_ptr &operator=(std::nullptr_t) {
    _ptr = nullptr;
    return *this;
  }

  // Only if Space == global_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S == access::address_space::global_space>
                * = nullptr>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::global_buffer,
               access::placeholder::false_t>
          a)
      : _ptr{reinterpret_cast<void *>(a.get_pointer().get())} {}

  // Only if Space == local_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<S == access::address_space::local_space>
                * = nullptr>
  HIPSYCL_KERNEL_TARGET
  multi_ptr(accessor<ElementType, dimensions, Mode, access::target::local,
                     access::placeholder::false_t>
                a)
      : _ptr{reinterpret_cast<void *>(a.get_pointer().get())} {}

  // Only if Space == constant_space
  template <typename ElementType, int dimensions, access::mode Mode,
            access::address_space S = Space,
            typename std::enable_if_t<
                S == access::address_space::constant_space> * = nullptr>
  HIPSYCL_KERNEL_TARGET multi_ptr(
      accessor<ElementType, dimensions, Mode, access::target::constant_buffer,
               access::placeholder::false_t>
          a)
      : _ptr{reinterpret_cast<void *>(a.get_pointer().get())} {}

  // Returns the underlying OpenCL C pointer
  HIPSYCL_UNIVERSAL_TARGET
  pointer_t get() const { return _ptr; }
  // TODO: Spec requires this to be implicit but that causes ambiguous overload
  // errors for operator==
  HIPSYCL_UNIVERSAL_TARGET
  operator void *() const { return _ptr; }

  // Explicit conversion to a multi_ptr<ElementType>
  template <typename ElementType>
  HIPSYCL_UNIVERSAL_TARGET explicit
  operator multi_ptr<ElementType, Space>() const {
    return multi_ptr<ElementType, Space>{reinterpret_cast<ElementType *>(_ptr)};
  }

  ACPP_MULTIPTR_MULTIPTR_COMP
  ACPP_MULTIPTR_NULLPTR_COMP

private:
  void *_ptr;
};

// Template specialization aliases for different pointer address spaces
template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using global_ptr =
    multi_ptr<ElementType, access::address_space::global_space, IsDecorated>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using local_ptr =
    multi_ptr<ElementType, access::address_space::local_space, IsDecorated>;

// Deprecated in SYCL 2020
template <typename ElementType>
using constant_ptr =
    multi_ptr<ElementType, access::address_space::constant_space,
              access::decorated::legacy>;

template <typename ElementType,
          access::decorated IsDecorated = access::decorated::legacy>
using private_ptr =
    multi_ptr<ElementType, access::address_space::private_space, IsDecorated>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes non-decorated pointer while keeping the
// address space information internally.
template <typename ElementType>
using raw_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::no>;

template <typename ElementType>
using raw_local_ptr = multi_ptr<ElementType, access::address_space::local_space,
                                access::decorated::no>;

template <typename ElementType>
using raw_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::no>;

// Template specialization aliases for different pointer address spaces.
// The interface exposes decorated pointer.
template <typename ElementType>
using decorated_global_ptr =
    multi_ptr<ElementType, access::address_space::global_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_local_ptr =
    multi_ptr<ElementType, access::address_space::local_space,
              access::decorated::yes>;

template <typename ElementType>
using decorated_private_ptr =
    multi_ptr<ElementType, access::address_space::private_space,
              access::decorated::yes>;

template <access::address_space Space, access::decorated DecorateAddress,
          typename ElementType>
HIPSYCL_UNIVERSAL_TARGET multi_ptr<ElementType, Space, DecorateAddress>
address_space_cast(ElementType *ptr) {
  return multi_ptr<ElementType, Space, DecorateAddress>{ptr};
}

// Deprecated, address_space_cast should be used instead.
template <typename ElementType, access::address_space Space,
          access::decorated DecorateAddress = access::decorated::legacy>
[[deprecated]] HIPSYCL_UNIVERSAL_TARGET
    multi_ptr<ElementType, Space, DecorateAddress>
    make_ptr(ElementType *ptr) {
  return address_space_cast<Space, DecorateAddress, ElementType>(ptr);
}

// Deduction guides
template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(
    accessor<T, Dimensions, access_mode::read, target::device, IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::global_space,
                 access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(
    accessor<T, Dimensions, access_mode::write, target::device, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read_write, target::device,
                   IsPlaceholder>)
    -> multi_ptr<T, access::address_space::global_space, access::decorated::no>;

template <typename T, int Dimensions, access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, access_mode::read, target::constant_buffer,
                   IsPlaceholder>)
    -> multi_ptr<const T, access::address_space::constant_space,
                 access::decorated::no>;

template <typename T, int Dimensions, access_mode Mode,
          access::placeholder IsPlaceholder>
multi_ptr(accessor<T, Dimensions, Mode, target::local, IsPlaceholder>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

template <typename T, int Dimensions>
multi_ptr(local_accessor<T, Dimensions>)
    -> multi_ptr<T, access::address_space::local_space, access::decorated::no>;

#undef ACPP_MULTIPTR_ARITHMETIC_OPS
#undef ACPP_MULTIPTR_MULTIPTR_COMP
#undef ACPP_MULTIPTR_NULLPTR_COMP
#undef ACPP_MULTIPTR_MULTIPTR_COMP

} // namespace sycl
} // namespace hipsycl

#endif
