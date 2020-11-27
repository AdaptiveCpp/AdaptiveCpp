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


#ifndef HIPSYCL_RANGE_HPP
#define HIPSYCL_RANGE_HPP

#include <type_traits>
#include <array>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "id.hpp"
#include "detail/device_array.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace sycl {

template <int dimensions = 1>
class range {
public:
  HIPSYCL_UNIVERSAL_TARGET
  range()
    : _data{}
  {}

  /* The following constructor is only available in the range class specialization where:
dimensions==1 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 1>>
  HIPSYCL_UNIVERSAL_TARGET
  range(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the range class specialization where:
dimensions==2 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 2>>
  HIPSYCL_UNIVERSAL_TARGET
  range(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the range class specialization where:
dimensions==3 */
  template<int D = dimensions,
           typename = std::enable_if_t<D == 3>>
  HIPSYCL_UNIVERSAL_TARGET
  range(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  /* -- common interface members -- */

  friend bool operator==(const range<dimensions>& lhs, const range<dimensions>& rhs){
    return lhs._data == rhs._data;
  }

  friend bool operator!=(const range<dimensions>& lhs, const range<dimensions>& rhs){
    return !(lhs == rhs);
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t get(int dimension) const {
    return _data[dimension];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t &operator[](int dimension) {
    return _data[dimension];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t operator[](int dimension) const {
    return _data[dimension];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t size() const {
    size_t result = 1;
    for(const auto x : _data)
      result *= x;
    return result;
  }

  // Implementation of id<dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend range<dimensions> operator op(const range<dimensions> &lhs, \
                                       const range<dimensions> &rhs) { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(lhs._data[i] op rhs._data[i]); \
    return result; \
  }

  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(+)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(-)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(*)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(/)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(%)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(<<)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(>>)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(&)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(|)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(^)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(&&)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(||)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(<)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(>)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(<=)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(>=)

#define HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend range<dimensions> operator op(const range<dimensions> &lhs, \
                                       const std::size_t &rhs) { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(lhs._data[i] op rhs); \
    return result; \
  }

  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(+)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(-)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(*)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(/)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(%)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<<)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>>)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(&)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(|)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(^)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(&&)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(||)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<=)
  HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>=)


  // Implementation of id<dimensions> &operatorOP(const id<dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define HIPSYCL_RANGE_BINARY_OP_IN_PLACE(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend range<dimensions>& operator op(range<dimensions> &lhs, \
                                 const range<dimensions> &rhs) { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      lhs._data[i] op rhs._data[i]; \
    return lhs; \
  }

  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(+=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(-=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(*=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(/=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(%=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(<<=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(>>=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(&=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(|=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE(^=)

#define HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend range<dimensions>& operator op(range<dimensions> &lhs, const std::size_t &rhs) { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      lhs._data[i] op rhs; \
    return lhs; \
  }

  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(+=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(-=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(*=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(/=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(%=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(<<=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(>>=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(&=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(|=)
  HIPSYCL_RANGE_BINARY_OP_IN_PLACE_SIZE_T(^=)


  #define HIPSYCL_RANGE_BINARY_OP_SIZE_T(op) \
  HIPSYCL_UNIVERSAL_TARGET \
  friend range<dimensions> operator op(const std::size_t &lhs, const range<dimensions> &rhs) { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result[i] = lhs op rhs[i]; \
    return result; \
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(+)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(-)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(*)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(/)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(%)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(<<)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(>>)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(&)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(|)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(^)

private:
  detail::device_array<size_t, dimensions> _data;

};

// deduction guides
range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

namespace detail {
namespace range {

HIPSYCL_UNIVERSAL_TARGET
inline sycl::range<2> omit_first_dimension(const sycl::range<3>& r)
{
  return sycl::range<2>{r.get(1), r.get(2)};
}

HIPSYCL_UNIVERSAL_TARGET
inline sycl::range<1> omit_first_dimension(const sycl::range<2>& r)
{
  return sycl::range<1>{r.get(1)};
}

template <int dimsOut, int dimsIn>
HIPSYCL_UNIVERSAL_TARGET
sycl::range<dimsOut> range_cast(const sycl::range<dimsIn>& other)
{
  sycl::range<dimsOut> result;
  for(size_t o = 0; o < dimsOut; ++o) {
    result[o] = o < dimsIn ? other[o] : 1;
  }
  return result;
}

}
}

} // sycl
} // hipsycl

#endif
