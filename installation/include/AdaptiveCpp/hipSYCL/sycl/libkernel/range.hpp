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

template <int Dimensions = 1>
class range {
public:
  static constexpr int dimensions = Dimensions;

  ACPP_UNIVERSAL_TARGET
  range()
    : _data{}
  {}

  /* The following constructor is only available in the range class specialization where:
Dimensions==1 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 1>>
  ACPP_UNIVERSAL_TARGET
  range(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the range class specialization where:
Dimensions==2 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 2>>
  ACPP_UNIVERSAL_TARGET
  range(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the range class specialization where:
Dimensions==3 */
  template<int D = Dimensions,
           typename = std::enable_if_t<D == 3>>
  ACPP_UNIVERSAL_TARGET
  range(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  /* -- common interface members -- */

  friend bool operator==(const range<Dimensions>& lhs, const range<Dimensions>& rhs){
    return lhs._data == rhs._data;
  }

  friend bool operator!=(const range<Dimensions>& lhs, const range<Dimensions>& rhs){
    return !(lhs == rhs);
  }

  ACPP_UNIVERSAL_TARGET
  size_t get(int dimension) const {
    return _data[dimension];
  }

  ACPP_UNIVERSAL_TARGET
  size_t &operator[](int dimension) {
    return _data[dimension];
  }

  ACPP_UNIVERSAL_TARGET
  size_t operator[](int dimension) const {
    return _data[dimension];
  }

  ACPP_UNIVERSAL_TARGET
  size_t size() const {
    // loop peel to help uniformity analysis
    size_t result = _data[0];
    for(int i = 1; i < Dimensions; ++i)
      result *= _data[i];
    return result;
  }

  // Implementation of id<Dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define HIPSYCL_RANGE_BINARY_OP_OUT_OF_PLACE(op) \
  ACPP_UNIVERSAL_TARGET \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs, \
                                       const range<Dimensions> &rhs) { \
    /* loop peel to help uniformity analysis */ \
    range<Dimensions> result; \
    result._data[0] = static_cast<std::size_t>(lhs._data[0] op rhs._data[0]); \
    for(std::size_t i = 1; i < Dimensions; ++i) \
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
  ACPP_UNIVERSAL_TARGET \
  friend range<Dimensions> operator op(const range<Dimensions> &lhs, \
                                       const std::size_t &rhs) { \
    /* loop peel to help uniformity analysis */ \
    range<Dimensions> result; \
    result._data[0] = static_cast<std::size_t>(lhs._data[0] op rhs); \
    for(std::size_t i = 1; i < Dimensions; ++i) \
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


  // Implementation of id<Dimensions> &operatorOP(const id<Dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define HIPSYCL_RANGE_BINARY_OP_IN_PLACE(op) \
  ACPP_UNIVERSAL_TARGET \
  friend range<Dimensions>& operator op(range<Dimensions> &lhs, \
                                 const range<Dimensions> &rhs) { \
    /* loop peel to help uniformity analysis */ \
    lhs._data[0] op rhs._data[0]; \
    for(std::size_t i = 1; i < Dimensions; ++i) \
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
  ACPP_UNIVERSAL_TARGET \
  friend range<Dimensions>& operator op(range<Dimensions> &lhs, const std::size_t &rhs) { \
    /* loop peel to help uniformity analysis */ \
    lhs._data[0] op rhs; \
    for(std::size_t i = 1; i < Dimensions; ++i) \
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
  ACPP_UNIVERSAL_TARGET \
  friend range<Dimensions> operator op(const std::size_t &lhs, const range<Dimensions> &rhs) { \
    /* loop peel to help uniformity analysis */ \
    range<Dimensions> result; \
    result[0] = lhs op rhs[0]; \
    for(std::size_t i = 1; i < Dimensions; ++i) \
      result[i] = lhs op rhs[i]; \
    return result; \
  }

  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
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
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(&&)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(||)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(<)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(>)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(<=)
  HIPSYCL_RANGE_BINARY_OP_SIZE_T(>=)

  // Unary operators +,-
  friend range operator+(const range& rhs) {
    return rhs;
  }

  friend range operator-(const range& rhs) {
    return -1*rhs;
  }

  // Prefix ++ and --
  friend range operator++(range &rhs) {
    rhs += 1;
    return rhs;
  }

  friend range operator--(range &rhs) {
    rhs -= 1;
    return rhs;
  }

  // Postfix ++ and --
  friend range operator++(range &rhs, int) {
    auto old = rhs;
    rhs += 1;
    return old;
  }

  friend range operator--(range &rhs, int) {
    auto old = rhs;
    rhs -= 1;
    return old;
  }

private:
  detail::device_array<size_t, Dimensions> _data;

};

// deduction guides
range(size_t) -> range<1>;
range(size_t, size_t) -> range<2>;
range(size_t, size_t, size_t) -> range<3>;

namespace detail {
namespace range {

ACPP_UNIVERSAL_TARGET
inline sycl::range<2> omit_first_dimension(const sycl::range<3>& r)
{
  return sycl::range<2>{r.get(1), r.get(2)};
}

ACPP_UNIVERSAL_TARGET
inline sycl::range<1> omit_first_dimension(const sycl::range<2>& r)
{
  return sycl::range<1>{r.get(1)};
}

template <int dimsOut, int dimsIn>
ACPP_UNIVERSAL_TARGET
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
