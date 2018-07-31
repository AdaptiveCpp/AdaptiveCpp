/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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


#ifndef SYCU_ID_HPP
#define SYCU_ID_HPP

#include <array>
#include <cassert>
#include <type_traits>

template<std::size_t dimensions>
class range;

template<std::size_t dimensions>
class item;

namespace cl {
namespace sycl {

template <int dimensions = 1>
struct id {

  __host__ __device__
  id()
    : _data{}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==1 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 1>>
  __host__ __device__
  id(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==2 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 2>>
  __host__ __device__
  id(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==3 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 3>>
  __host__ __device__
  id(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim1, dim2}
  {}

  __host__ __device__
  id(const id<dimensions>& other) = default;

  __host__ __device__
  id(const range<dimensions> &range) {
    for(std::size_t i = 0; i < dimensions; ++i)
      this->_data[i] = range[i];
  }

  __host__ __device__
  id(const item<dimensions> &item) {
    for(std::size_t i = 0; i < dimensions; ++i)
      this->_data[i] = item[i];
  }

  __host__ __device__
  size_t get(int dimension) const {
    return this->_data[dimension];
  }

  __host__ __device__
  size_t& operator[](int dimension) const {
    // Spec requires that this method should be const, but return
    // a non-const reference...
    return const_cast<size_t&>(this->_data[dimension]);
  }

  // Implementation of id<dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define SYCU_ID_BINARY_OP_OUT_OF_PLACE(op) \
  __host__ __device__  \
  id<dimensions> operator op(const id<dimensions> &rhs) const { \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(_data[i] op rhs._data[i]); \
    return result; \
  }

  SYCU_ID_BINARY_OP_OUT_OF_PLACE(+)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(-)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(*)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(/)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(%)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(<<)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(>>)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(&)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(|)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(^)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(&&)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(||)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(<)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(>)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(<=)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE(>=)

#define SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(op) \
  __host__ __device__ \
  id<dimensions> operator op(const std::size_t &rhs) const { \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(_data[i] op rhs); \
    return result; \
  }

  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(+)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(-)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(*)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(/)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(%)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<<)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>>)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(&)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(|)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(^)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(&&)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(||)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(<=)
  SYCU_ID_BINARY_OP_OUT_OF_PLACE_SIZE_T(>=)


  // Implementation of id<dimensions> &operatorOP(const id<dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define SYCU_ID_BINARY_OP_IN_PLACE(op) \
  __host__ __device__ \
  id<dimensions>& operator op(const id<dimensions> &rhs) const { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      _data[i] op rhs._data[i]; \
    return *this; \
  }

  SYCU_ID_BINARY_OP_IN_PLACE(+=)
  SYCU_ID_BINARY_OP_IN_PLACE(-=)
  SYCU_ID_BINARY_OP_IN_PLACE(*=)
  SYCU_ID_BINARY_OP_IN_PLACE(/=)
  SYCU_ID_BINARY_OP_IN_PLACE(%=)
  SYCU_ID_BINARY_OP_IN_PLACE(<<=)
  SYCU_ID_BINARY_OP_IN_PLACE(>>=)
  SYCU_ID_BINARY_OP_IN_PLACE(&=)
  SYCU_ID_BINARY_OP_IN_PLACE(|=)
  SYCU_ID_BINARY_OP_IN_PLACE(^=)

#define SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(op) \
  __host__ __device__ \
  id<dimensions>& operator op(const std::size_t &rhs) const { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      _data[i] op rhs; \
    return *this; \
  }

  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(+=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(-=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(*=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(/=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(%=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(<<=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(>>=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(&=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(|=)
  SYCU_ID_BINARY_OP_IN_PLACE_SIZE_T(^=)
private:
  std::array<std::size_t, dimensions> _data;
};

#define SYCU_ID_BINARY_OP_SIZE_T(op) \
  template<int dimensions> \
  __host__ __device__ \
  id<dimensions> operator op(const size_t &lhs, const id<dimensions> &rhs) { \
    id<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result[i] = lhs op rhs[i]; \
    return result; \
  }

// OP is: +, -, *, /, %, <<, >>, &, |, ˆ
SYCU_ID_BINARY_OP_SIZE_T(+)
SYCU_ID_BINARY_OP_SIZE_T(-)
SYCU_ID_BINARY_OP_SIZE_T(*)
SYCU_ID_BINARY_OP_SIZE_T(/)
SYCU_ID_BINARY_OP_SIZE_T(%)
SYCU_ID_BINARY_OP_SIZE_T(<<)
SYCU_ID_BINARY_OP_SIZE_T(>>)
SYCU_ID_BINARY_OP_SIZE_T(&)
SYCU_ID_BINARY_OP_SIZE_T(|)
SYCU_ID_BINARY_OP_SIZE_T(^)

} // namespace sycl
} // namespace cl

#endif
