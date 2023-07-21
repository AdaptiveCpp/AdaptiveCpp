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

#ifndef HIPSYCL_MARRAY_HPP
#define HIPSYCL_MARRAY_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "detail/device_array.hpp"
#include "half.hpp"

namespace hipsycl {
namespace sycl {

template <typename DataT, std::size_t NumElements> class marray {
private:
  detail::device_array<DataT, NumElements> _data;
  
  template<class T>
  static constexpr std::size_t count_elements() {
    if constexpr (std::is_scalar_v<T>)
      return 1;
    else
      return T::size();
  }

  template<class T>
  constexpr void init_offset(std::size_t offset, const T& x) {
    if constexpr(std::is_scalar_v<T>) {
      _data[offset] = x;
    } else {
      for(std::size_t i = 0; i < _data.size(); ++i)
      _data[offset + i] = x[i];  
    }
  }

  template<typename ArgT>
  constexpr void initialize_from_arg(int& current_offset, const ArgT& arg) {
    init_offset(current_offset, arg);
    current_offset += count_elements<ArgT>();
  }

  struct not_convertible_to_scalar {};
  static constexpr auto get_scalar_conversion_type() {
    if constexpr(NumElements == 1)
      return std::size_t{};
    else
      return not_convertible_to_scalar {};
  }

  using scalar_conversion_type = decltype(get_scalar_conversion_type());
public:
  using value_type = DataT;
  using reference = DataT&;
  using const_reference = const DataT&;
  using iterator = DataT*;
  using const_iterator = const DataT*;

  marray()
  : marray{DataT{}} {}

  explicit constexpr marray(const DataT& arg) {
    for(std::size_t i = 0; i < NumElements; ++i) {
      _data[i] = arg;
    }
  }

  template <typename... ArgTN>
  constexpr marray(const ArgTN&... args) {
    constexpr std::size_t num_args = (count_elements<ArgTN>() + ...);
    static_assert(num_args == NumElements,
                  "Incorrect number of marray constructor arguments");
    int current_offset = 0;
    (initialize_from_arg(current_offset, args), ...);
  }

  constexpr marray(const marray<DataT, NumElements>& rhs) = default;
  constexpr marray(marray<DataT, NumElements>&& rhs) = default;

  // Available only when: NumElements == 1
  operator scalar_conversion_type() const {
    return _data[0];
  }

  static constexpr std::size_t size() noexcept {
    return NumElements;
  }

  // subscript operator
  reference operator[](std::size_t index) {
    return _data[index];
  }
  const_reference operator[](std::size_t index) const {
    return _data[index];
  }

  marray& operator=(const marray<DataT, NumElements>& rhs) = default;
  marray& operator=(const DataT& rhs) {
    for(int i = 0; i < NumElements; ++i) {
      _data[i] = rhs;
    }
    return *this;
  }

  // iterator functions
  // These rely on device_array having contiguous memory layout --
  // let's hope this is the case despite the registerization
  // optimizations that it does for small arrays.. This is pretty fishy..
  iterator begin() {
    return &_data[0];
  }
  const_iterator begin() const {
    return &_data[0];
  }

  iterator end() {
    return begin() + size();
  }
  const_iterator end() const {
    return begin() + size();
  }

  // OP is: +, -, *, /, %
  /* If OP is %, available only when: DataT != float && DataT != double && DataT
   * != half. */
#define HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(op, T)                   \
  friend marray<T, NumElements> operator op(const marray &lhs,                 \
                                            const marray &rhs) {               \
    marray<T, NumElements> result;                                             \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs[i] op rhs[i];                                            \
    }                                                                          \
    return result;                                                             \
  }
#define HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(op, T)                   \
  friend marray<T, NumElements> operator op(const marray &lhs, const T &rhs) { \
    marray<T, NumElements> result;                                             \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }
#define HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(op, T)                   \
  friend marray<T, NumElements> operator op(const T &lhs, const marray &rhs) { \
    marray<T, NumElements> result;                                             \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs op rhs[i];                                               \
    }                                                                          \
    return result;                                                             \
  }

  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(+, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(-, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(*, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(/, DataT)
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(%, t)

  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(+, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(-, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(*, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(/, DataT)
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(%, t)

#define HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(op, T)                  \
  friend marray &operator op(marray &lhs, const marray<T, NumElements> &rhs) { \
    for (int i = 0; i < NumElements; ++i) {                                    \
      lhs[i] op rhs[i];                                                        \
    }                                                                          \
    return lhs;                                                                \
  }

#define HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(op, T)                  \
  friend marray &operator op(marray &lhs, const T &rhs) {                      \
    for (int i = 0; i < NumElements; ++i) {                                    \
      lhs[i] op rhs;                                                           \
    }                                                                          \
    return lhs;                                                                \
  }

  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(+=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(-=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(*=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(/=, DataT)
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(%=, t)

  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(+=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(-=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(*=, DataT)
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(/=, DataT)
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(%=, t)

  // OP is prefix ++, --
  friend marray& operator++(marray& rhs) {
    for(int i = 0; i < NumElements; ++i)
      ++(rhs._data[i]);
    return rhs;
  }

  friend marray& operator--(marray& rhs) {
    for(int i = 0; i < NumElements; ++i)
      --(rhs._data[i]);
    return rhs;
  }

  // OP is postfix ++, --
  friend marray operator++(marray& lhs, int) {
    marray<DataT,NumElements> old = lhs;

    for(int i = 0; i < NumElements; ++i)
      ++(lhs._data[i]);

    return old;
  }

  friend marray operator--(marray& lhs, int) {
    marray<DataT,NumElements> old = lhs;

    for(int i = 0; i < NumElements; ++i)
      --(lhs._data[i]);

    return old;
  }

  // OP is unary +, -
  friend marray<DataT, NumElements> operator+(const marray& v) {
    return v;
  }

  friend marray<DataT,NumElements> operator-(const marray& v) {
    marray<DataT,NumElements> result;
    for(int i = 0; i < NumElements; ++i) {
      result._data[i] = -(v._data[i]);
    }
    return result;
  }

  // OP is: &, |, ^
  /* Available only when: DataT != float && DataT != double && DataT != half. */
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(&, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(|, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(^, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(&, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(|, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(^, t)

  // OP is: &=, |=, ^=
  /* Available only when: DataT != float && DataT != double && DataT != half. */
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(&=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(|=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(^=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(&=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(|=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(^=, t)

#define HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(op)                            \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const marray &rhs) {            \
    marray<bool, NumElements> result;                                          \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs[i] op rhs[i];                                            \
    }                                                                          \
    return result;                                                             \
  }

#define HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(op)                            \
  friend marray<bool, NumElements> operator op(const marray &lhs,              \
                                               const DataT &rhs) {             \
    marray<bool, NumElements> result;                                          \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs[i] op rhs;                                               \
    }                                                                          \
    return result;                                                             \
  }

#define HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(op)                            \
  friend marray<bool, NumElements> operator op(const DataT &lhs,               \
                                               const marray &rhs) {            \
    marray<bool, NumElements> result;                                          \
    for (int i = 0; i < NumElements; ++i) {                                    \
      result[i] = lhs op rhs[i];                                               \
    }                                                                          \
    return result;                                                             \
  }
    // OP is: &&, ||

  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(&&)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(||)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(&&)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(||)

  // OP is: <<, >>
  /* Available only when: DataT != float && DataT != double && DataT != half.
    */
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(>>, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_MARRAY(<<, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(>>, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_MARRAY_SCALAR(<<, t)

  // OP is: <<=, >>=
  /* Available only when: DataT != float && DataT != double && DataT != half.
    */

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(>>=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_MARRAY(<<=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(>>=, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_MARRAY_OP_MARRAY_SCALAR(<<=, t)

  // OP is: ==, !=, <, >, <=, >=
  
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(==)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(!=)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(<)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(>)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(<=)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_MARRAY(>=)

  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(==)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(!=)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(<)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(>)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(<=)
  HIPSYCL_LOGICAL_MARRAY_OP_MARRAY_SCALAR(>=)


  /* Available only when: DataT != float && DataT != double && DataT != half.
    */
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  friend marray<t, NumElements> operator~(const marray& v) {
    marray<t, NumElements> result;
    for(int i = 0; i < NumElements; ++i) {
      result._data[i] = ~(v._data[i]);
    }
    return result;
  }


  // OP is: +, -, *, /, %
  /* operator% is only available when: DataT != float && DataT != double &&
    * DataT != half. */
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(+, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(-, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(*, DataT)
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(/, DataT)
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(%, t)

  // OP is: &, |, ^
  /* Available only when: DataT != float && DataT != double
  && DataT != half. */
  
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(&, t)
  
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(|, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(^, t)

  // OP is: &&, ||
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(&&)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(||)

  // OP is: <<, >>
  /* Available only when: DataT != float && DataT != double && DataT != half.
    */
  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(<<, t)

  template <typename t = DataT,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_MARRAY_OP_SCALAR_MARRAY(>>, t)

  // OP is: ==, !=, <, >, <=, >=

  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(==)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(!=)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(<)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(>)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(<=)
  HIPSYCL_LOGICAL_MARRAY_OP_SCALAR_MARRAY(>=)


  friend marray<bool, NumElements> operator!(const marray& v) {
    marray<bool, NumElements> result;
    for(int i = 0; i < NumElements; ++i){
      result[i] = !v._data[i];
    }
    return result;
  }
};

template <class T, class... U,
          class = std::enable_if_t<(std::is_same<T, U>::value && ...)>>
marray(T, U...) -> marray<T, sizeof...(U) + 1>;

#define MARRAY_TYPE_ALIASES(type, storage_type, elements)   \
  using m##type##elements = marray<storage_type, elements>;

#define MARRAY_TYPE_ALIASES_ALL(type, storage_type)   \
  MARRAY_TYPE_ALIASES(type, storage_type, 2);         \
  MARRAY_TYPE_ALIASES(type, storage_type, 3);         \
  MARRAY_TYPE_ALIASES(type, storage_type, 4);         \
  MARRAY_TYPE_ALIASES(type, storage_type, 8);         \
  MARRAY_TYPE_ALIASES(type, storage_type, 16);

  MARRAY_TYPE_ALIASES_ALL(char, int8_t);
  MARRAY_TYPE_ALIASES_ALL(uchar, uint8_t);
  MARRAY_TYPE_ALIASES_ALL(short, int16_t);
  MARRAY_TYPE_ALIASES_ALL(ushort, uint16_t);
  MARRAY_TYPE_ALIASES_ALL(int, int32_t);
  MARRAY_TYPE_ALIASES_ALL(uint, uint32_t);
  MARRAY_TYPE_ALIASES_ALL(long, int64_t);
  MARRAY_TYPE_ALIASES_ALL(ulong, uint64_t);
  MARRAY_TYPE_ALIASES_ALL(half, half);
  MARRAY_TYPE_ALIASES_ALL(float, float);
  MARRAY_TYPE_ALIASES_ALL(double, double);
  MARRAY_TYPE_ALIASES_ALL(bool, bool);
}
}

#endif
