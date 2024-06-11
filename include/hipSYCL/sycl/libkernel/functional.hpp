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


#ifndef HIPSYCL_SYCL_FUNCTIONAL_HPP
#define HIPSYCL_SYCL_FUNCTIONAL_HPP

#include "backend.hpp"
#include "half.hpp"
#include <limits>
#include <type_traits>

namespace hipsycl {
namespace sycl {

// TODO We might want to alias these to std:: types?
template <typename T = void> struct plus {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <> struct plus<void> {
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T = void> struct multiplies {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template<> struct multiplies<void> {
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T = void> struct bit_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template<> struct bit_and <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template <typename T = void> struct bit_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template<> struct bit_or <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template <typename T = void> struct bit_xor {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template<> struct bit_xor <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <typename T = void> struct logical_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template<> struct logical_and <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template <typename T = void> struct logical_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template<> struct logical_or <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template <typename T = void> struct minimum {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template<> struct minimum <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template <typename T = void> struct maximum {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x > y) ? x : y; }
};

template<> struct maximum <void>{
  template<class T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x > y) ? x : y; }
};

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity {};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity : public std::false_type {};

template <class U, class AccumulatorT>
struct has_known_identity<plus<U>, AccumulatorT> {
  static constexpr bool value =
      std::is_arithmetic_v<AccumulatorT> ||
      std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>;
};

template <class U, class AccumulatorT>
struct has_known_identity<multiplies<U>, AccumulatorT> {
  static constexpr bool value =
      std::is_arithmetic_v<AccumulatorT> ||
      std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>;
};

template <class U, class AccumulatorT>
struct has_known_identity<bit_and<U>, AccumulatorT> {
  static constexpr bool value = std::is_integral_v<AccumulatorT>;
};

template <class U, class AccumulatorT>
struct has_known_identity<bit_or<U>, AccumulatorT> {
  static constexpr bool value = std::is_integral_v<AccumulatorT>;
};

template <class U, class AccumulatorT>
struct has_known_identity<bit_xor<U>, AccumulatorT> {
  static constexpr bool value = std::is_integral_v<AccumulatorT>;
};

template <class U, class AccumulatorT>
struct has_known_identity<logical_and<U>, AccumulatorT> {
  static constexpr bool value = std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>;
};

template <class U, class AccumulatorT>
struct has_known_identity<logical_or<U>, AccumulatorT> {
  static constexpr bool value = std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>;
};

template <class U, class AccumulatorT>
struct has_known_identity<minimum<U>, AccumulatorT> {
  static constexpr bool value =
      std::is_integral_v<AccumulatorT> ||
      std::is_floating_point_v<AccumulatorT> ||
      std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>;
};

template <class U, class AccumulatorT>
struct has_known_identity<maximum<U>, AccumulatorT> {
  static constexpr bool value =
      std::is_integral_v<AccumulatorT> ||
      std::is_floating_point_v<AccumulatorT> ||
      std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>;
};

template<class U, class AccumulatorT>
struct known_identity<plus<U>, AccumulatorT> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity<multiplies<U>, AccumulatorT> {
  static constexpr AccumulatorT value = AccumulatorT{1};
};

template<class U, class AccumulatorT>
struct known_identity<bit_and<U>, AccumulatorT> {
  static constexpr AccumulatorT value = ~AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity<bit_or<U>, AccumulatorT> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity<bit_xor<U>, AccumulatorT> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity<logical_and<U>, AccumulatorT> {
  static constexpr AccumulatorT value = true;
};

template<class U, class AccumulatorT>
struct known_identity<logical_or<U>, AccumulatorT> {
  static constexpr AccumulatorT value = false;
};

template<class U, class AccumulatorT>
struct known_identity<minimum<U>, AccumulatorT> {
private:
  static constexpr auto get() {
    if constexpr(std::is_integral_v<AccumulatorT>)
      return std::numeric_limits<AccumulatorT>::max();
    else return std::numeric_limits<AccumulatorT>::infinity();
  }
public:
  static constexpr AccumulatorT value = get();
};

template<class U, class AccumulatorT>
struct known_identity<maximum<U>, AccumulatorT> {
private:
  static constexpr auto get() {
    if constexpr(std::is_integral_v<AccumulatorT>)
      return std::numeric_limits<AccumulatorT>::lowest();
    else return -std::numeric_limits<AccumulatorT>::infinity();
  }
public:
  static constexpr AccumulatorT value = get();
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;



} // namespace sycl
}

#endif
