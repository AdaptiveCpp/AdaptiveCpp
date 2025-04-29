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
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <> struct plus<void> {
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T = void> struct multiplies {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template<> struct multiplies<void> {
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T = void> struct bit_and {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template<> struct bit_and <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template <typename T = void> struct bit_or {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template<> struct bit_or <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template <typename T = void> struct bit_xor {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template<> struct bit_xor <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <typename T = void> struct logical_and {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template<> struct logical_and <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template <typename T = void> struct logical_or {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template<> struct logical_or <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template <typename T = void> struct minimum {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template<> struct minimum <void>{
  template<class T>
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template <typename T = void> struct maximum {
  ACPP_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x > y) ? x : y; }
};

template<> struct maximum <void>{
  template<class T>
  ACPP_KERNEL_TARGET
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
