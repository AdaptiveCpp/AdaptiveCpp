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

template <typename, std::size_t>
class marray;

template <typename, int, class>
class vec;

// TODO We might want to alias these to std:: types?
template <typename T = void>
using plus = std::plus<T>;
template <typename T = void>
using multiplies = std::multiplies<T>;
template <typename T = void>
using bit_and = std::bit_and<T>;
template <typename T = void>
using bit_or = std::bit_or<T>;
template <typename T = void>
using bit_xor = std::bit_xor<T>;

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

namespace detail {
template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct known_identity_impl;
}

template <typename BinaryOperation, typename AccumulatorT>
struct known_identity : detail::known_identity_impl<BinaryOperation, std::decay_t<AccumulatorT>> {};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v =
    known_identity<BinaryOperation, AccumulatorT>::value;

namespace detail {
template <typename BinaryOperation, typename AccumulatorT, typename = void>
struct has_known_identity_impl : std::false_type {};

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity_impl<BinaryOperation, AccumulatorT, std::void_t<decltype(known_identity<BinaryOperation, AccumulatorT>::value)>> : std::true_type {};
}

template <typename BinaryOperation, typename AccumulatorT>
struct has_known_identity : detail::has_known_identity_impl<BinaryOperation, AccumulatorT> {};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v =
    has_known_identity<BinaryOperation, AccumulatorT>::value;

namespace detail {
template <typename BinaryOperation, typename AccumulatorT, typename >
struct known_identity_impl{};

template<class U, class AccumulatorT>
struct known_identity_impl<plus<U>, AccumulatorT,
  std::enable_if_t<std::is_arithmetic_v<AccumulatorT> ||
    std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>>> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity_impl<multiplies<U>, AccumulatorT,
  std::enable_if_t<std::is_arithmetic_v<AccumulatorT> ||
    std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>>> {
  static constexpr AccumulatorT value = AccumulatorT{1};
};

template<class U, class AccumulatorT>
struct known_identity_impl<bit_and<U>, AccumulatorT,
  std::enable_if_t<std::is_integral_v<AccumulatorT>>> {
  static constexpr AccumulatorT value = ~AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity_impl<bit_or<U>, AccumulatorT,
  std::enable_if_t<std::is_integral_v<AccumulatorT>>> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity_impl<bit_xor<U>, AccumulatorT,
  std::enable_if_t<std::is_integral_v<AccumulatorT>>> {
  static constexpr AccumulatorT value = AccumulatorT{};
};

template<class U, class AccumulatorT>
struct known_identity_impl<logical_and<U>, AccumulatorT,
  std::enable_if_t<std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>>> {
  static constexpr AccumulatorT value = true;
};

template<class U, class AccumulatorT>
struct known_identity_impl<logical_or<U>, AccumulatorT,
  std::enable_if_t<std::is_same_v<std::remove_cv_t<AccumulatorT>, bool>>> {
  static constexpr AccumulatorT value = false;
};

template<class U, class AccumulatorT>
struct known_identity_impl<minimum<U>, AccumulatorT,
  std::enable_if_t<std::is_integral_v<AccumulatorT>>> {
  static constexpr AccumulatorT value = std::numeric_limits<AccumulatorT>::max();
};

template<class U, class AccumulatorT>
struct known_identity_impl<minimum<U>, AccumulatorT,
  std::enable_if_t<std::is_floating_point_v<AccumulatorT> ||
    std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>>> {
  static constexpr AccumulatorT value = std::numeric_limits<AccumulatorT>::infinity();
};

template<class U, class AccumulatorT>
struct known_identity_impl<maximum<U>, AccumulatorT,
  std::enable_if_t<std::is_integral_v<AccumulatorT>>> {
  static constexpr AccumulatorT value = std::numeric_limits<AccumulatorT>::lowest();
};

template<class U, class AccumulatorT>
struct known_identity_impl<maximum<U>, AccumulatorT,
  std::enable_if_t<std::is_floating_point_v<AccumulatorT> ||
    std::is_same_v<std::remove_cv_t<AccumulatorT>, sycl::half>>> {
  static constexpr AccumulatorT value = -std::numeric_limits<AccumulatorT>::infinity();
};

template <typename BinaryOperation, typename AccumulatorT, std::size_t NumElements>
struct known_identity_impl<BinaryOperation, marray<AccumulatorT, NumElements>,
  std::enable_if_t<has_known_identity_v<BinaryOperation, AccumulatorT>>> {
  static constexpr auto value = marray<AccumulatorT, NumElements>(known_identity_v<BinaryOperation, AccumulatorT>);
};

template <typename BinaryOperation, typename AccumulatorT, int N, class VectorStorage>
struct known_identity_impl<BinaryOperation, vec<AccumulatorT, N, VectorStorage>,
  std::enable_if_t<has_known_identity_v<BinaryOperation, AccumulatorT>>> {
  static constexpr auto value = vec<AccumulatorT, N, VectorStorage>(known_identity_v<BinaryOperation, AccumulatorT>);
};
}


} // namespace sycl
}

#endif
