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
#include <type_traits>

#include "hipSYCL/runtime/util.hpp"

#include "backend.hpp"
#include "detail/device_array.hpp"


namespace hipsycl {
namespace sycl {

template<int dimensions>
class range;

template<int dimensions, bool with_offset>
struct item;

template <int dimensions = 1>
struct id {

  HIPSYCL_UNIVERSAL_TARGET
  id()
    : _data{}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==1 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 1>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==2 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 2>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==3 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 3>>
  HIPSYCL_UNIVERSAL_TARGET
  id(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  /* -- common interface members -- */

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator==(const id<dimensions>& lhs, const id<dimensions>& rhs){
    return lhs._data == rhs._data;
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator!=(const id<dimensions>& lhs, const id<dimensions>& rhs){
    return lhs._data != rhs._data;
  }

  HIPSYCL_UNIVERSAL_TARGET
  id(const range<dimensions> &range) {
    for(std::size_t i = 0; i < dimensions; ++i)
      this->_data[i] = range[i];
  }

  template<bool with_offset>
  HIPSYCL_UNIVERSAL_TARGET
  id(const item<dimensions, with_offset> &item) {
    for(std::size_t i = 0; i < dimensions; ++i)
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
/*
  template <int D = dimensions, typename = std::enable_if_t<D == 1>>
  HIPSYCL_UNIVERSAL_TARGET
  operator size_t() const {
    return this->_data[0];
  }
  */
  // Implementation of id<dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET  \
  friend id<dimensions> operator op(const id<dimensions> &lhs, const id<dimensions> &rhs) { \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
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

#define HIPSYCL_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend id<dimensions> operator op(const id<dimensions> &lhs, \
                             const std::size_t &rhs){ \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(lhs._data[i] op rhs); \
    return result; \
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


  // Implementation of id<dimensions> &operatorOP(const id<dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define HIPSYCL_ID_BINARY_OP_IN_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend id<dimensions>& operator op(id<dimensions> &lhs, const id<dimensions> &rhs) { \
    for(std::size_t i = 0; i < dimensions; ++i) \
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

#define HIPSYCL_ID_BINARY_OP_IN_PLACE_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend id<dimensions>& operator op(id<dimensions> &lhs, const std::size_t &rhs) { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      lhs._data[i] op rhs; \
    return lhs; \
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

#define HIPSYCL_ID_BINARY_OP_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend id<dimensions> operator op(const size_t &lhs, const id<dimensions> &rhs) { \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result[i] = lhs op rhs[i]; \
    return result; \
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ
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

private:
  detail::device_array<std::size_t, dimensions> _data;
};

// Deduction guides
id(size_t) -> id<1>;
id(size_t, size_t) -> id<2>;
id(size_t, size_t, size_t) -> id<3>;

namespace detail {
namespace id{

template<int dimensions>
HIPSYCL_UNIVERSAL_TARGET
inline sycl::id<dimensions> construct_from_first_n(size_t x, size_t y, size_t z);

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
