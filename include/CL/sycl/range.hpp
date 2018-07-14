#ifndef SYCU_RANGE_HPP
#define SYCU_RANGE_HPP

#include <type_traits>
#include <array>

namespace cl {
namespace sycl {

template <std::size_t dimensions = 1>
class range {
public:

  /* The following constructor is only available in the range class specialization where:
dimensions==1 */
  template<typename = std::enable_if_t<dimensions == 1>>
  range(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the range class specialization where:
dimensions==2 */
  template<typename = std::enable_if_t<dimensions == 2>>
  range(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the range class specialization where:
dimensions==3 */
  template<typename = std::enable_if_t<dimensions == 3>>
  range(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim0, dim1, dim2}
  {}

  /* -- common interface members -- */

  std::size_t get(int dimension) const {
    return _data[dimension];
  }

  std::size_t &operator[](int dimension) {
    return _data[dimension];
  }

  std::size_t size() const {
    std::size_t result = 1;
    for(const auto x : _data)
      result *= x;
    return result;
  }

  // Implementation of id<dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(op) \
  range<dimensions> operator op(const range<dimensions> &rhs) const { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(_data[i] op rhs._data[i]); \
    return result; \
  }

  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(+)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(-)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(*)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(/)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(%)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(<<)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(>>)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(&)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(|)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(^)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(&&)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(||)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(<)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(>)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(<=)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE(>=)

#define SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(op) \
  range<dimensions> operator op(const std::size_t &rhs) const { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result._data[i] = static_cast<std::size_t>(_data[i] op rhs); \
    return result; \
  }

  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(+)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(-)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(*)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(/)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(%)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<<)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>>)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(&)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(|)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(^)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(&&)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(||)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(<=)
  SYCU_RANGE_BINARY_OP_OUT_OF_PLACE_SIZE_T(>=)


  // Implementation of id<dimensions> &operatorOP(const id<dimensions> &rhs);
  // OP is: +=, -=, *=, /=, %=, <<=, >>=, &=, |=, ˆ=
#define SYCU_RANGE_BINARY_OP_IN_PLACE(op) \
  range<dimensions>& operator op(const range<dimensions> &rhs) const { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      _data[i] op rhs._data[i]; \
    return *this; \
  }

  SYCU_RANGE_BINARY_OP_IN_PLACE(+=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(-=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(*=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(/=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(%=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(<<=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(>>=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(&=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(|=)
  SYCU_RANGE_BINARY_OP_IN_PLACE(^=)

#define SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(op) \
  range<dimensions>& operator op(const std::size_t &rhs) const { \
    for(std::size_t i = 0; i < dimensions; ++i) \
      _data[i] op rhs; \
    return *this; \
  }

  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(+=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(-=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(*=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(/=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(%=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(<<=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(>>=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(&=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(|=)
  SYCU_RANGE_BINARY_OP_IN_PLACE_SIZE_T(^=)


private:
  std::array<std::size_t, dimensions> _data;

};


#define SYCU_RANGE_BINARY_OP_SIZE_T(op) \
  template<int dimensions> \
  range<dimensions> operator op(const std::size_t &lhs, const id<dimensions> &rhs) { \
    range<dimensions> result; \
    for(std::size_t i = 0; i < dimensions; ++i) \
      result[i] = lhs op rhs[i]; \
    return result; \
  }

// OP is: +, -, *, /, %, <<, >>, &, |, ˆ
SYCU_RANGE_BINARY_OP_SIZE_T(+)
SYCU_RANGE_BINARY_OP_SIZE_T(-)
SYCU_RANGE_BINARY_OP_SIZE_T(*)
SYCU_RANGE_BINARY_OP_SIZE_T(/)
SYCU_RANGE_BINARY_OP_SIZE_T(%)
SYCU_RANGE_BINARY_OP_SIZE_T(<<)
SYCU_RANGE_BINARY_OP_SIZE_T(>>)
SYCU_RANGE_BINARY_OP_SIZE_T(&)
SYCU_RANGE_BINARY_OP_SIZE_T(|)
SYCU_RANGE_BINARY_OP_SIZE_T(^)

} // sycl
} // cl

#endif
