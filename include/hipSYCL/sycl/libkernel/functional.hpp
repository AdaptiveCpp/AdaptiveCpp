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

namespace hipsycl {
namespace sycl {

template <typename T = void> struct plus {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <> struct plus<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x + y; }
};

template <typename T = void> struct multiplies {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template <> struct multiplies<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x * y; }
};

template <typename T = void> struct bit_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template <> struct bit_and<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x & y; }
};

template <typename T = void> struct bit_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x | y; }
};

template <> struct bit_or<void> {
    template<typename T>
    HIPSYCL_KERNEL_TARGET
    T operator()(const T &x, const T &y) const { return x | y; }
};

template <typename T = void> struct bit_xor {
    HIPSYCL_KERNEL_TARGET
    T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <> struct bit_xor<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return x ^ y; }
};

template <typename T = void> struct logical_and {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template <> struct logical_and<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x && y); }
};

template <typename T = void> struct logical_or {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template <> struct logical_or<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return static_cast<T>(x || y); }
};

template <typename T = void> struct minimum {
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template <> struct minimum<void> {
    template<typename T>
    HIPSYCL_KERNEL_TARGET
    T operator()(const T &x, const T &y) const { return (x < y) ? x : y; }
};

template <typename T = void> struct maximum {
    HIPSYCL_KERNEL_TARGET
    T operator()(const T &x, const T &y) const { return (x > y) ? x : y; }
};

template <> struct maximum<void> {
  template<typename T>
  HIPSYCL_KERNEL_TARGET
  T operator()(const T &x, const T &y) const { return (x > y) ? x : y; }
};


namespace detail {

template<typename BinaryOperation, typename AccumulatorT, typename Enable = void>
struct known_identity_trait {
    static constexpr bool has_known_identity = false; \
};

#define HIPSYCL_DEFINE_IDENTITY(op, cond, identity) \
    template<typename T, typename U> \
    struct known_identity_trait<op<T>, U, std::enable_if_t<cond>> { \
        static constexpr bool has_known_identity = true; \
        inline static constexpr T known_identity = (identity); \
    }; \
    template<typename T> \
    struct known_identity_trait<op<void>, T, std::enable_if_t<cond>> { \
        inline static constexpr bool has_known_identity = true; \
        inline static constexpr T known_identity = (identity); \
    }

template<typename T, typename Enable=void>
struct minmax_identity {
    inline static constexpr T max_id = std::numeric_limits<T>::lowest();
    inline static constexpr T min_id = std::numeric_limits<T>::max();
};

template<typename T>
struct minmax_identity<T, std::enable_if_t<std::numeric_limits<T>::has_infinity>> {
    inline static constexpr T max_id = -std::numeric_limits<T>::infinity();
    inline static constexpr T min_id = std::numeric_limits<T>::infinity();
};

// TODO is_arithmetic implicitly covers the current pseudo half = ushort type, resolve once half is implemented
HIPSYCL_DEFINE_IDENTITY(plus, std::is_arithmetic_v<T>, T{});
HIPSYCL_DEFINE_IDENTITY(multiplies, std::is_arithmetic_v<T>, T{1});
HIPSYCL_DEFINE_IDENTITY(bit_or, std::is_integral_v<T>, T{});
HIPSYCL_DEFINE_IDENTITY(bit_and, std::is_integral_v<T>, ~T{});
HIPSYCL_DEFINE_IDENTITY(bit_xor, std::is_integral_v<T>, T{});
HIPSYCL_DEFINE_IDENTITY(logical_or, (std::is_same_v<T, bool>), false);
HIPSYCL_DEFINE_IDENTITY(logical_and, (std::is_same_v<T, bool>), true);
HIPSYCL_DEFINE_IDENTITY(minimum, std::is_arithmetic_v<T>, minmax_identity<T>::min_id);
HIPSYCL_DEFINE_IDENTITY(maximum, std::is_arithmetic_v<T>, minmax_identity<T>::max_id);

#undef HIPSYCL_DEFINE_IDENTITY

}

template<typename BinaryOperation, typename AccumulatorT>
struct known_identity {
    static constexpr AccumulatorT value = detail::known_identity_trait<
        BinaryOperation, AccumulatorT>::known_identity;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr AccumulatorT known_identity_v = known_identity<BinaryOperation, AccumulatorT>::value;

template<typename BinaryOperation, typename AccumulatorT>
struct has_known_identity {
    static constexpr bool value = detail::known_identity_trait<
        BinaryOperation, AccumulatorT>::has_known_identity;
};

template <typename BinaryOperation, typename AccumulatorT>
inline constexpr bool has_known_identity_v = has_known_identity<BinaryOperation, AccumulatorT>::value;

} // namespace sycl
}

#endif
