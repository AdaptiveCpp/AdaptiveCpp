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
#ifndef HIPSYCL_ATOMIC_REF_HPP
#define HIPSYCL_ATOMIC_REF_HPP

#include <cstdint>
#include <type_traits>

#include "memory.hpp"
#include "atomic_builtins.hpp"

namespace hipsycl {
namespace sycl {

template <memory_order ReadModifyWriteOrder>
struct memory_order_traits;

template <>
struct memory_order_traits<memory_order::relaxed> {
  static constexpr memory_order read_order = memory_order::relaxed;
  static constexpr memory_order write_order = memory_order::relaxed;
};

template <>
struct memory_order_traits<memory_order::acq_rel> {
  static constexpr memory_order read_order = memory_order::acquire;
  static constexpr memory_order write_order = memory_order::release;
};

template <>
struct memory_order_traits<memory_order::seq_cst> {
  static constexpr memory_order read_order = memory_order::seq_cst;
  static constexpr memory_order write_order = memory_order::seq_cst;
};


inline void atomic_fence(memory_order order, memory_scope scope) {
  detail::__acpp_atomic_fence(order, scope);
}


template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space Space = access::address_space::generic_space,
          typename Enable = void>
class atomic_ref {
public:
  static_assert(std::is_same_v<T, int> || std::is_same_v<T, unsigned int> ||
                    std::is_same_v<T, long> ||
                    std::is_same_v<T, unsigned long> ||
                    std::is_same_v<T, long long> ||
                    std::is_same_v<T, unsigned long long> ||
                    std::is_same_v<T, float> || std::is_same_v<T, double> ||
                    std::is_pointer_v<T>,
                "Invalid data type for atomic_ref");
};

template <typename Integral,
          memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space Space>
class atomic_ref<Integral, DefaultOrder, DefaultScope, Space, std::enable_if_t<std::is_integral_v<Integral>>> {
public:

  static_assert(Space == access::address_space::generic_space ||
                    Space == access::address_space::global_space ||
                    Space == access::address_space::local_space,
                "Invalid address space for atomic_ref");

  using value_type = Integral;
  using difference_type = value_type;

  static constexpr std::size_t required_alignment = alignof(Integral);
  // TODO
  static constexpr bool is_always_lock_free = true;

  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept {
    // TODO
    return true;
  }

  explicit atomic_ref(Integral& x)
  : _ptr{&x} {}

  atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&) = delete;

  void store(Integral operand,
    memory_order order = default_write_order,
    memory_scope scope = default_scope) const noexcept {
    detail::__acpp_atomic_store<Space>(_ptr, operand, order, scope);
  }

  Integral operator=(Integral desired) const noexcept {
    store(desired);
    return desired;
  }

  Integral load(memory_order order = default_read_order,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_load<Space>(_ptr, order, scope);
  }

  operator Integral() const noexcept {
    return load();
  }

  Integral exchange(Integral operand,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_exchange<Space>(_ptr, operand, order, scope);
  }

  bool compare_exchange_weak(Integral &expected, Integral desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_compare_exchange_weak<Space>(
        _ptr, expected, desired, success, failure, scope);
  }

  bool
  compare_exchange_weak(Integral &expected, Integral desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool compare_exchange_strong(Integral &expected, Integral desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_compare_exchange_strong<Space>(
        _ptr, expected, desired, success, failure, scope);
  }

  bool compare_exchange_strong(Integral &expected, Integral desired,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  Integral fetch_add(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_add<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_sub(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_sub<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_and(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_and<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_or(Integral operand,
                    memory_order order = default_read_modify_write_order,
                    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_or<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_xor(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_xor<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_min(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_min<Space>(_ptr, operand, order, scope);
  }

  Integral fetch_max(Integral operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_max<Space>(_ptr, operand, order, scope);
  }

  Integral operator++(int) const noexcept {
    return fetch_add(Integral{1});
  }

  Integral operator--(int) const noexcept {
    return fetch_sub(Integral{1});
  }

  Integral operator++() const noexcept {
    return fetch_add(Integral{1}) + 1;
  }

  Integral operator--() const noexcept {
    return fetch_sub(Integral{1}) - 1;
  }

  Integral operator+=(Integral op) const noexcept {
    return fetch_add(op) + op;
  }

  Integral operator-=(Integral op) const noexcept {
    return fetch_sub(op) - op;
  }

  Integral operator&=(Integral op) const noexcept {
    return fetch_and(op) & op;
  }

  Integral operator|=(Integral op) const noexcept {
    return fetch_or(op) | op;
  }

  Integral operator^=(Integral op) const noexcept {
    return fetch_xor(op) ^ op;
  }

private:
  Integral* _ptr;
};

template <typename Floating, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space Space>
class atomic_ref<Floating, DefaultOrder, DefaultScope, Space, std::enable_if_t<std::is_floating_point_v<Floating>>> {
public:

  static_assert(Space == access::address_space::generic_space ||
                    Space == access::address_space::global_space ||
                    Space == access::address_space::local_space,
                "Invalid address space for atomic_ref");

  using value_type = Floating;
  using difference_type = value_type;

  static constexpr std::size_t required_alignment = alignof(Floating);
  // TODO
  static constexpr bool is_always_lock_free = true;

  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept {
    // TODO
    return true;
  }

  explicit atomic_ref(Floating& x)
  : _ptr{&x} {}

  atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&) = delete;

  void store(Floating operand,
    memory_order order = default_write_order,
    memory_scope scope = default_scope) const noexcept {
    detail::__acpp_atomic_store<Space>(_ptr, operand, order, scope);
  }

  Floating operator=(Floating desired) const noexcept {
    store(desired);
    return desired;
  }

  Floating load(memory_order order = default_read_order,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_load<Space>(_ptr, order, scope);
  }

  operator Floating() const noexcept {
    return load();
  }

  Floating exchange(Floating operand,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_exchange<Space>(_ptr, operand, order, scope);
  }

  bool compare_exchange_weak(Floating &expected, Floating desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_compare_exchange_weak<Space>(
        _ptr, expected, desired, success, failure, scope);
  }

  bool
  compare_exchange_weak(Floating &expected, Floating desired,
                        memory_order order = default_read_modify_write_order,
                        memory_scope scope = default_scope) const noexcept {
    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool compare_exchange_strong(Floating &expected, Floating desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_compare_exchange_strong<Space>(
        _ptr, expected, desired, success, failure, scope);
  }

  bool compare_exchange_strong(Floating &expected, Floating desired,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  Floating fetch_add(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_add<Space>(_ptr, operand, order,
                                                     scope);
  }

  Floating fetch_sub(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_sub<Space>(_ptr, operand, order,
                                                     scope);
  }

  Floating fetch_min(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_min<Space>(_ptr, operand, order,
                                                     scope);
  }

  Floating fetch_max(Floating operand,
                     memory_order order = default_read_modify_write_order,
                     memory_scope scope = default_scope) const noexcept {
    return detail::__acpp_atomic_fetch_max<Space>(_ptr, operand, order,
                                                     scope);
  }

  Floating operator+=(Floating op) const noexcept {
    return fetch_add(op) + op;
  }

  Floating operator-=(Floating op) const noexcept {
    return fetch_sub(op) - op;
  }

private:
  Floating* _ptr;
};

template <typename T, memory_order DefaultOrder, memory_scope DefaultScope,
          access::address_space Space>
class atomic_ref<T *, DefaultOrder, DefaultScope, Space> {

  static std::intptr_t ptr_to_int(T *p) {
    return reinterpret_cast<std::intptr_t>(p);
  }

  static T *int_to_ptr(std::intptr_t i) {
    return reinterpret_cast<T *>(i);
  }

  static std::intptr_t &ptr_ref_to_int_ref(T *&p) {
    T **pp = &p;
    return *reinterpret_cast<std::intptr_t *>(pp);
  }

public:
  using value_type = T*;
  using difference_type = std::ptrdiff_t;
  static constexpr std::size_t required_alignment = alignof(T*);
  // TODO
  static constexpr bool is_always_lock_free = true;

  static constexpr memory_order default_read_order =
      memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
      memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order = DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  bool is_lock_free() const noexcept {
    // TODO
    return true;
  }

  explicit atomic_ref(T*& val) {
    _ptr = reinterpret_cast<std::intptr_t*>(&val);
  }

  atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&) = delete;

  void store(T* operand,
    memory_order order = default_write_order,
    memory_scope scope = default_scope) const noexcept {
    detail::__acpp_atomic_store<Space>(
        _ptr, ptr_to_int(operand), order, scope);
  }

  T* operator=(T* desired) const noexcept {
    store(desired);
    return desired;
  }

  T* load(memory_order order = default_read_order,
    memory_scope scope = default_scope) const noexcept {
    std::intptr_t v =
        detail::__acpp_atomic_load<Space>(_ptr, order, scope);
    return int_to_ptr(v);
  }

  operator T*() const noexcept {
    return load();
  }

  T* exchange(T* operand,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    std::intptr_t v = detail::__acpp_atomic_exchange<Space>(
        _ptr, ptr_to_int(operand), order, scope);
    return int_to_ptr(v);
  }

  bool compare_exchange_weak(T* &expected, T* desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {

    std::intptr_t desired_v = ptr_to_int(desired);
    std::intptr_t& expected_v = ptr_ref_to_int_ref(expected);

    return detail::__acpp_atomic_compare_exchange_weak<Space>(
        _ptr, expected_v, desired_v, success, failure, scope);
  }

  bool compare_exchange_weak(T* &expected, T* desired,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {

    return compare_exchange_weak(expected, desired, order, order, scope);
  }

  bool compare_exchange_strong(T* &expected, T* desired,
    memory_order success,
    memory_order failure,
    memory_scope scope = default_scope) const noexcept {
    
    std::intptr_t desired_v = ptr_to_int(desired);
    std::intptr_t& expected_v = ptr_ref_to_int_ref(expected);

    return detail::__acpp_atomic_compare_exchange_strong<Space>(
        _ptr, expected_v, desired_v, success, failure, scope);
  }

  bool compare_exchange_strong(T* &expected, T* desired,
    memory_order order = default_read_modify_write_order,
    memory_scope scope = default_scope) const noexcept {
    
    return compare_exchange_strong(expected, desired, order, order, scope);
  }

  T *fetch_add(difference_type x,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {

    return int_to_ptr(detail::__acpp_atomic_fetch_add<Space>(
        _ptr, static_cast<std::intptr_t>(x * sizeof(T)), order, scope));
  }

  T *fetch_sub(difference_type x,
               memory_order order = default_read_modify_write_order,
               memory_scope scope = default_scope) const noexcept {

    return int_to_ptr(detail::__acpp_atomic_fetch_sub<Space>(
        _ptr, static_cast<std::intptr_t>(x * sizeof(T)), order, scope));
  }

  T* operator++(int) const noexcept {
    return fetch_add(1);
  }

  T* operator--(int) const noexcept {
    return fetch_sub(1);
  }

  T* operator++() const noexcept {
    return fetch_add(1) + 1;
  }

  T* operator--() const noexcept {
    return fetch_sub(1) - 1;
  }

  T* operator+=(difference_type x) const noexcept {
    return fetch_add(x) + x;
  }

  T* operator-=(difference_type x) const noexcept {
    return fetch_sub(x) - x;
  }
private:
  std::intptr_t* _ptr;
};
}
}

#endif
