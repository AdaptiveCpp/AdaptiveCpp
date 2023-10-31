/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018,2019 Aksel Alpay
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


#ifndef HIPSYCL_ID_HPP
#define HIPSYCL_ID_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include "hipSYCL/runtime/util.hpp"

#include "backend.hpp"
#include "detail/device_array.hpp"


namespace hipsycl {
namespace sycl {

template<int Dimensions>
class range;

template<int Dimensions, bool with_offset>
struct item;

template <int Dimensions = 1>
struct id {
private:
  struct not_convertible_to_scalar {};

  static constexpr auto get_scalar_conversion_type() {
    if constexpr(Dimensions == 1)
      return std::size_t{};
    else
      return not_convertible_to_scalar {};
  }

  using scalar_conversion_type = decltype(get_scalar_conversion_type());

public:
  static constexpr int dimensions = Dimensions;

  HIPSYCL_UNIVERSAL_TARGET
  id()
    : _data{}
  {}

  /* The following constructor is only available in the id class
   * specialization where: Dimensions==1 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 1>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the id class
   * specialization where: Dimensions==2 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 2>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the id class
   * specialization where: Dimensions==3 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 3>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  /* -- common interface members -- */

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const id<Dimensions>& lhs, const id<Dimensions>& rhs){
    return lhs._data == rhs._data;
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const id<Dimensions>& lhs, const id<Dimensions>& rhs){
    return lhs._data != rhs._data;
  }

  HIPSYCL_UNIVERSAL_TARGET
  id(const range<Dimensions> &range) {
    /* loop peel to help uniformity analysis */ \
    this->_data[0] = range[0];
    for(std::size_t i = 1; i < Dimensions; ++i)
      this->_data[i] = range[i];
  }

  template<bool with_offset>
  HIPSYCL_UNIVERSAL_TARGET
  id(const item<Dimensions, with_offset> &item) {
    /* loop peel to help uniformity analysis */ \
    this->_data[0] = item.get_id(0);
    for(std::size_t i = 1; i < Dimensions; ++i)
      this->_data[i] = item.get_id(i);
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t get(int dimension) const {
    return this->_data[dimension];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t& operator[](int dimension) {
    return this->_data[dimension];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t operator[](int dimension) const {
    return this->_data[dimension];
  }

  // We cannot use enable_if since the involved templates would
  // prevent implicit type conversion to other integer types.
  HIPSYCL_UNIVERSAL_TARGET
  operator scalar_conversion_type () const {
    return this->_data[0];
  }

  // Implementation of id<Dimensions> operatorOP(const id &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET  \
  friend id<Dimensions> operator op(const id<Dimensions> &lhs, const id<Dimensions> &rhs) { \
    id<Dimensions> result; \
    /* loop peel to help uniformity analysis */ \
    result._data[0] = static_cast<std::size_t>(lhs._data[0] op rhs._data[0]); \
    for(std::size_t i = 1; i < Dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(lhs._data[i] op rhs._data[i]); \
    return result; \
  }

  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(+)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(-)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(*)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(/)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(%)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(<<)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(>>)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(&)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(|)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(^)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(&&)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(||)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(<)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(>)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(<=)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(>=)

#define HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(op)                           \
  template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>          \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend id<Dimensions> operator op(const id<Dimensions> &lhs,                 \
                                    const T &rhs) {                            \
    id<Dimensions> result;                                                     \
    /* loop peel to help uniformity analysis */ \
    result._data[0] = static_cast<T>(lhs._data[0] op rhs);                                                                           \
    for (std::size_t i = 1; i < Dimensions; ++i)                               \
      result._data[i] = static_cast<T>(lhs._data[i] op rhs);                   \
    return result;                                                             \
  }                                                                            \
  /* Dedicated overload for range to avoid operator ambiguity due to */        \
  /* implicit conversion to size_t                                   */        \
  template <int D = Dimensions, std::enable_if_t<D == 1, int> = 0>             \
  HIPSYCL_UNIVERSAL_TARGET friend id<Dimensions> operator op(                  \
      const id<Dimensions> &lhs, const range<Dimensions> &rhs) {               \
    return lhs op rhs[0];                                                      \
  }

  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(+)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(-)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(*)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(/)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(%)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<<)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>>)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(&)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(|)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(^)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(&&)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(||)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<=)
  HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>=)


  // Implementation of id<Dimensions> &operatorOP(const id<Dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define HIPSYCL_ID_BINARY_OP_IN_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend id<Dimensions>& operator op(id<Dimensions> &lhs, const id<Dimensions> &rhs) { \
    /* loop peel to help uniformity analysis */ \
    lhs._data[0] op rhs._data[0]; \
    for(std::size_t i = 1; i < Dimensions; ++i) \
      lhs._data[i] op rhs._data[i]; \
    return lhs; \
  }

  HIPSYCL_ID_BINARY_OP_IN_PLACE(+=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(-=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(*=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(/=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(%=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(<<=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(>>=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(&=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(|=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE(^=)

#define HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(op)                          \
  template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>     \
  HIPSYCL_UNIVERSAL_TARGET                                                \
  friend id<Dimensions>& operator op(id<Dimensions> &lhs, const T &rhs) { \
    /* loop peel to help uniformity analysis */ \
    lhs._data[0] op rhs;                                                  \
    for(std::size_t i = 1; i < Dimensions; ++i)                           \
      lhs._data[i] op rhs;                                                \
    return lhs;                                                           \
  }

  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(+=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(-=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(*=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(/=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(%=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(<<=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(>>=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(&=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(|=)
  HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(^=)

#define HIPSYCL_ID_BINARY_OP_SIZE_T(op)                                        \
  template<class T, std::enable_if_t<std::is_integral_v<T>, int> = 0>          \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend id<Dimensions> operator op(const T &lhs, const id<Dimensions> &rhs) { \
    id<Dimensions> result;                                                     \
    /* loop peel to help uniformity analysis */ \
    result[0] = lhs op rhs[0];                                                 \
    for(std::size_t i = 1; i < Dimensions; ++i)                                \
      result[i] = lhs op rhs[i];                                               \
    return result;                                                             \
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
  HIPSYCL_ID_BINARY_OP_SIZE_T(+)
  HIPSYCL_ID_BINARY_OP_SIZE_T(-)
  HIPSYCL_ID_BINARY_OP_SIZE_T(*)
  HIPSYCL_ID_BINARY_OP_SIZE_T(/)
  HIPSYCL_ID_BINARY_OP_SIZE_T(%)
  HIPSYCL_ID_BINARY_OP_SIZE_T(<<)
  HIPSYCL_ID_BINARY_OP_SIZE_T(>>)
  HIPSYCL_ID_BINARY_OP_SIZE_T(&)
  HIPSYCL_ID_BINARY_OP_SIZE_T(|)
  HIPSYCL_ID_BINARY_OP_SIZE_T(^)
  HIPSYCL_ID_BINARY_OP_SIZE_T(&&)
  HIPSYCL_ID_BINARY_OP_SIZE_T(||)
  HIPSYCL_ID_BINARY_OP_SIZE_T(<)
  HIPSYCL_ID_BINARY_OP_SIZE_T(>)
  HIPSYCL_ID_BINARY_OP_SIZE_T(<=)
  HIPSYCL_ID_BINARY_OP_SIZE_T(>=)

  // Unary operators +,-
  friend id operator+(const id& rhs) {
    return rhs;
  }

  friend id operator-(const id& rhs) {
    return -1*rhs;
  }

  // Prefix ++ and --
  friend id operator++(id &rhs) {
    rhs += 1;
    return rhs;
  }

  friend id operator--(id &rhs) {
    rhs -= 1;
    return rhs;
  }

  // Postfix ++ and --
  friend id operator++(id &rhs, int) {
    auto old = rhs;
    rhs += 1;
    return old;
  }

  friend id operator--(id &rhs, int) {
    auto old = rhs;
    rhs -= 1;
    return old;
  }

private:
  detail::device_array<std::size_t, Dimensions> _data;
};

// Deduction guides
id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

namespace detail {
namespace id{

template<int Dimensions>
HIPSYCL_UNIVERSAL_TARGET
inline sycl::id<Dimensions> construct_from_first_n(size_t x, size_t y, size_t z);

template<>
HIPSYCL_UNIVERSAL_TARGET
inline sycl::id<3> construct_from_first_n(size_t x, size_t y, size_t z)
{ return sycl::id<3>{x,y,z}; }

template<>
HIPSYCL_UNIVERSAL_TARGET
inline sycl::id<2> construct_from_first_n(size_t x, size_t y, size_t z)
{ return sycl::id<2>{x,y}; }

template<>
HIPSYCL_UNIVERSAL_TARGET
inline sycl::id<1> construct_from_first_n(size_t x, size_t y, size_t z)
{ return sycl::id<1>{x}; }


} // namespace id
} // namespace detail

} // namespace sycl
} // namespace hipsycl

#endif
