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

template <std::size_t dimensions = 1>
struct id {

  id()
    : _data{}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==1 */
  template<typename = std::enable_if_t<dimensions == 1>>
  id(size_t dim0)
    : _data{dim0}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==2 */
  template<typename = std::enable_if_t<dimensions == 2>>
  id(size_t dim0, size_t dim1)
    : _data{dim0, dim1}
  {}

  /* The following constructor is only available in the id class
   * specialization where: dimensions==3 */
  template<typename = std::enable_if_t<dimensions == 3>>
  id(size_t dim0, size_t dim1, size_t dim2)
    : _data{dim1, dim2}
  {}

  id(const id<dimensions>& other) = default;


  id(const range<dimensions> &range) {
    for(std::size_t i = 0; i < dimensions; ++i)
      this->_data[i] = range[i];
  }

  id(const item<dimensions> &item) {
    for(std::size_t i = 0; i < dimensions; ++i)
      this->_data[i] = item[i];
  }

  size_t get(int dimension) const {
    return this->_data[dimension];
  }

  size_t& operator[](int dimension) const {
    return this->_data[dimension];
  }

  // Implementation of id<dimensions> operatorOP(const size_t &rhs) const;
  // OP is: +, -, *, /, %, <<, >>, &, |, ˆ, &&, ||, <, >, <=, >=
#define SYCU_ID_BINARY_OP_OUT_OF_PLACE(op) \
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
