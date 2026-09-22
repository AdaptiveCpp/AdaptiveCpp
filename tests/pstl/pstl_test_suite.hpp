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

#ifndef HIPSYCL_PSTL_TEST_SUITE_HPP
#define HIPSYCL_PSTL_TEST_SUITE_HPP

#include <cstddef>
#include <iterator>

struct enable_unified_shared_memory {
  enable_unified_shared_memory() {
#ifndef __ACPP_STDPAR_ASSUME_SYSTEM_USM__
    hipsycl::stdpar::unified_shared_memory::pop_disabled();
#endif
  }

  ~enable_unified_shared_memory() {
#ifndef __ACPP_STDPAR_ASSUME_SYSTEM_USM__
    hipsycl::stdpar::unified_shared_memory::push_disabled();
#endif
  }
};


static thread_local int counter = 0;
struct non_trivial_copy {

  non_trivial_copy(){}

  non_trivial_copy(int val)
  : x{val} {}

  non_trivial_copy(const non_trivial_copy& other){
    x = other.x;
    __acpp_if_target_host(++counter;)
  }

  non_trivial_copy& operator=(const non_trivial_copy& other) {
    x = other.x;
    __acpp_if_target_host(++counter;)
    return *this;
  }

  friend bool operator==(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x == b.x;
  }

  friend bool operator!=(const non_trivial_copy &a, const non_trivial_copy &b) {
    return a.x != b.x;
  }

  int x;
};


static thread_local int move_counter = 0;
struct non_trivial_move {

  non_trivial_move(){}

  non_trivial_move(int val)
  : x{val} {}

  non_trivial_move(const non_trivial_move& other)
  : x{other.x} {}

  non_trivial_move& operator=(const non_trivial_move& other) {
    x = other.x;
    return *this;
  }

  non_trivial_move(non_trivial_move&& other) {
    x = std::move(other.x);
    __acpp_if_target_host(++move_counter;)
  }

  non_trivial_move& operator=(non_trivial_move&& other) {
    if (this != &other) {
      x = std::move(other.x);
      __acpp_if_target_host(++move_counter;)
    }
    return *this;
  }  

  friend bool operator==(const non_trivial_move &a, const non_trivial_move &b) {
    return a.x == b.x;
  }

  friend bool operator!=(const non_trivial_move &a, const non_trivial_move &b) {
    return a.x != b.x;
  }

  int x;
};


// Minimal forward iterator whose operator*() returns by value (no backing
// storage) -- regression test for copy()/move() requiring the memcpy
// fast-path's address-of expressions to be gated by `if constexpr` on
// is_contiguous<T>(), not a runtime `if` (which requires them to compile
// unconditionally for every iterator type). Do NOT replace with
// boost::counting_iterator: its operator*() returns a real reference, so
// it does not reproduce this bug.
template<class T>
struct counting_iterator {
  using value_type = T;
  using difference_type = std::ptrdiff_t;
  using pointer = void;
  using reference = T;
  using iterator_category = std::forward_iterator_tag;

  counting_iterator() = default;
  explicit counting_iterator(T value) : _value{value} {}

  T operator*() const { return _value; }
  counting_iterator& operator++() { ++_value; return *this; }
  counting_iterator operator++(int) { auto tmp = *this; ++(*this); return tmp; }

  friend bool operator==(const counting_iterator& a, const counting_iterator& b) { return a._value == b._value; }
  friend bool operator!=(const counting_iterator& a, const counting_iterator& b) { return a._value != b._value; }

  T _value{};
};

#endif
