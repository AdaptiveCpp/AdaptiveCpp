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
#ifndef HIPSYCL_MATH_BUILTINS_HPP
#define HIPSYCL_MATH_BUILTINS_HPP

#include <cstdint>
#include <type_traits>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/half.hpp"
#include "hipSYCL/sycl/libkernel/marray.hpp"
#include "hipSYCL/sycl/libkernel/multi_ptr.hpp"
#include "hipSYCL/sycl/libkernel/vec.hpp"
#include "hipSYCL/sycl/libkernel/builtin_interface.hpp"

#include "builtin_utils.hpp"

namespace hipsycl::sycl {
namespace detail {
  template <typename T>
  struct is_valid_math_scalar : is_in_type_set<float, double, half>::scalar_tester<T> {};

  template <typename T>
  struct is_valid_special_math_scalar : is_in_type_set<float>::scalar_tester<T> {};

  template <typename T>
  struct is_valid_math_nonscalar : std::bool_constant<
    is_valid_math_scalar<builtin_input_element_t<T>>::value &&
    builtin_input_is_nonscalar_v<T>
  > {};

  template <typename T>
  struct is_valid_special_math_nonscalar : std::bool_constant<
    is_valid_special_math_scalar<builtin_input_element_t<T>>::value &&
    builtin_input_is_nonscalar_v<T>
  > {};

  template <typename T, typename Ptr>
  struct is_math_valid_ptr : std::bool_constant<std::is_same_v<T*, Ptr>> {};

  template <typename T, typename ElementType, access::address_space Space, access::decorated DecorateAddress>
  struct is_math_valid_ptr<T, multi_ptr<ElementType, Space, DecorateAddress>> : std::bool_constant<
    (
      Space == access::address_space::global_space ||
      Space == access::address_space::local_space ||
      Space == access::address_space::private_space ||
      Space == access::address_space::private_space
    ) && std::is_same_v<T, ElementType>
  > {};

  template <typename T, typename Ptr>
  constexpr bool is_math_valid_ptr_v = is_math_valid_ptr<T, Ptr>::value;
}

#define HIPSYCL_MATH_BUILTIN_GENERATOR_UNARY_T(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN T builtin_name(T x) noexcept {                              \
    return builtin_func(x);                                                   \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_UNARY_T_RET_INT(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN int builtin_name(T x) noexcept {                                    \
    return builtin_func(x);                                                           \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_T(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN T builtin_name(T x, T y) noexcept {                            \
    return builtin_func(x, y);                                                   \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_TSCALAR HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_T

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_INT(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN T builtin_name(T x, int y) noexcept {                            \
    return builtin_func(x, y);                                                     \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_INT_SCALAR HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_INT

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_TPTR(T, builtin_name, builtin_func) \
  template<typename Ptr>                                                            \
  HIPSYCL_BUILTIN                                                                   \
  std::enable_if_t<                                                                 \
    detail::is_math_valid_ptr_v<T, Ptr>,                                            \
    T                                                                               \
  > builtin_name(T x, Ptr y) noexcept {                                             \
    T& iy = *y;                                                                     \
    return builtin_func(x, &iy);                                                    \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_BINARY_T_INTPTR(T, builtin_name, builtin_func) \
  template<typename Ptr>                                                              \
  HIPSYCL_BUILTIN                                                                     \
  std::enable_if_t<                                                                   \
    detail::is_math_valid_ptr_v<int, Ptr>,                                            \
    T                                                                                 \
  > builtin_name(T x, Ptr y) noexcept {                                               \
    int& iy = *y;                                                                     \
    return builtin_func(x, &iy);                                                      \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TERNARY_T(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN T builtin_name(T x, T y, T z) noexcept {                      \
    return builtin_func(x, y, z);                                               \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_UNARY_T(tester, builtin_name, builtin_func) \
  template<typename NonScalar1>                                                             \
  HIPSYCL_BUILTIN                                                                           \
  std::enable_if_t<                                                                         \
    detail::builtin_inputs_are_compatible_v<                                                \
      NonScalar1                                                                            \
    > &&                                                                                    \
    tester<NonScalar1>::value,                                                              \
    detail::builtin_input_return_t<NonScalar1>                                              \
  > builtin_name(NonScalar1 x) noexcept {                                                   \
    detail::builtin_input_return_t<NonScalar1> ret;                                         \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {       \
      ret[i] = builtin_func(x[i]);                                                          \
    }                                                                                       \
    return ret;                                                                             \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_UNARY_T_RET_INT(tester, builtin_name, builtin_func) \
  template<typename NonScalar1>                                                                     \
  HIPSYCL_BUILTIN                                                                                   \
  std::enable_if_t<                                                                                 \
    detail::builtin_inputs_are_compatible_v<                                                        \
      NonScalar1                                                                                    \
    > &&                                                                                            \
    tester<NonScalar1>::value,                                                                      \
    detail::builtin_input_intlike_t<NonScalar1>                                                     \
  > builtin_name(NonScalar1 x) noexcept {                                                           \
    detail::builtin_input_intlike_t<NonScalar1> ret;                                                \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {               \
      ret[i] = builtin_func(x[i]);                                                                  \
    }                                                                                               \
    return ret;                                                                                     \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_T(tester, builtin_name, builtin_func) \
  template<typename NonScalar1, typename NonScalar2>                                           \
  HIPSYCL_BUILTIN                                                                              \
  std::enable_if_t<                                                                            \
    detail::builtin_inputs_are_compatible_v<                                                   \
      NonScalar1, NonScalar2                                                                   \
    > &&                                                                                       \
    tester<NonScalar1>::value,                                                                 \
    detail::builtin_input_return_t<NonScalar1>                                                 \
  > builtin_name(NonScalar1 x, NonScalar2 y) noexcept {                                        \
    detail::builtin_input_return_t<NonScalar1> ret;                                            \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {          \
      ret[i] = builtin_func(x[i], y[i]);                                                       \
    }                                                                                          \
    return ret;                                                                                \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_TSCALAR(tester, builtin_name, builtin_func) \
  HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_T(tester, builtin_name, builtin_func)             \
  template<typename NonScalar1>                                                                      \
  HIPSYCL_BUILTIN                                                                                    \
  std::enable_if_t<                                                                                  \
    detail::builtin_inputs_are_compatible_v<                                                         \
      NonScalar1                                                                                     \
    > &&                                                                                             \
    tester<NonScalar1>::value,                                                                       \
    detail::builtin_input_return_t<NonScalar1>                                                       \
  > builtin_name(NonScalar1 x, typename NonScalar1::value_type y) noexcept {                         \
    detail::builtin_input_return_t<NonScalar1> ret;                                                  \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {                \
      ret[i] = builtin_func(x[i], y);                                                                \
    }                                                                                                \
    return ret;                                                                                      \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_INT(tester, builtin_name, builtin_func) \
  template<typename NonScalar1, typename NonScalar2>                                             \
  HIPSYCL_BUILTIN                                                                                \
  std::enable_if_t<                                                                              \
    detail::builtin_inputs_are_compatible_v<                                                     \
      NonScalar1                                                                                 \
    > &&                                                                                         \
    tester<NonScalar1>::value &&                                                                 \
    detail::builtin_inputs_are_compatible_v<                                                     \
      detail::builtin_input_intlike_t<NonScalar1>, NonScalar2                                    \
    >,                                                                                           \
    detail::builtin_input_return_t<NonScalar1>                                                   \
  > builtin_name(NonScalar1 x, NonScalar2 y) noexcept {                                          \
    detail::builtin_input_return_t<NonScalar1> ret;                                              \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {            \
      ret[i] = builtin_func(x[i], y[i]);                                                         \
    }                                                                                            \
    return ret;                                                                                  \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_INT_SCALAR(tester, builtin_name, builtin_func) \
  HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_INT(tester, builtin_name, builtin_func)              \
  template<typename NonScalar1>                                                                         \
  HIPSYCL_BUILTIN                                                                                       \
  std::enable_if_t<                                                                                     \
    detail::builtin_inputs_are_compatible_v<                                                            \
      NonScalar1                                                                                        \
    > &&                                                                                                \
    tester<NonScalar1>::value,                                                                          \
    detail::builtin_input_return_t<NonScalar1>                                                          \
  > builtin_name(NonScalar1 x, int y) noexcept {                                                        \
    detail::builtin_input_return_t<NonScalar1> ret;                                                     \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {                   \
      ret[i] = builtin_func(x[i], y);                                                                   \
    }                                                                                                   \
    return ret;                                                                                         \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_TPTR(tester, builtin_name, builtin_func) \
  template<typename NonScalar1, typename Ptr>                                                     \
  HIPSYCL_BUILTIN                                                                                 \
  std::enable_if_t<                                                                               \
    detail::builtin_inputs_are_compatible_v<                                                      \
      NonScalar1                                                                                  \
    > &&                                                                                          \
    tester<NonScalar1>::value &&                                                                  \
    detail::is_math_valid_ptr_v<                                                                  \
      detail::builtin_input_return_t<NonScalar1>, Ptr                                             \
    >,                                                                                            \
    detail::builtin_input_return_t<NonScalar1>                                                    \
  > builtin_name(NonScalar1 x, Ptr iptr) noexcept {                                               \
    detail::builtin_input_return_t<NonScalar1> ret;                                               \
    detail::builtin_input_return_t<NonScalar1>& y = *iptr;                                        \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {             \
      ret[i] = builtin_func(x[i], &y[i]);                                                         \
    }                                                                                             \
    return ret;                                                                                   \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_BINARY_T_INTPTR(tester, builtin_name, builtin_func) \
  template<typename NonScalar1, typename Ptr>                                                       \
  HIPSYCL_BUILTIN                                                                                   \
  std::enable_if_t<                                                                                 \
    detail::builtin_inputs_are_compatible_v<                                                        \
      NonScalar1                                                                                    \
    > &&                                                                                            \
    tester<NonScalar1>::value &&                                                                    \
    detail::is_math_valid_ptr_v<                                                                    \
      detail::builtin_input_intlike_t<NonScalar1>, Ptr                                              \
    >,                                                                                              \
    detail::builtin_input_return_t<NonScalar1>                                                      \
  > builtin_name(NonScalar1 x, Ptr iptr) noexcept {                                                 \
    detail::builtin_input_return_t<NonScalar1> ret;                                                 \
    detail::builtin_input_intlike_t<NonScalar1>& y = *iptr;                                         \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {               \
      ret[i] = builtin_func(x[i], &y[i]);                                                           \
    }                                                                                               \
    return ret;                                                                                     \
  }

#define HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_TERNARY_T(tester, builtin_name, builtin_func) \  
  template<typename NonScalar1, typename NonScalar2, typename NonScalar3>                     \
  HIPSYCL_BUILTIN                                                                             \
  std::enable_if_t<                                                                           \
    detail::builtin_inputs_are_compatible_v<                                                  \
      NonScalar1, NonScalar2, NonScalar3                                                      \
    > &&                                                                                      \
    tester<NonScalar1>::value,                                                                \
    detail::builtin_input_return_t<NonScalar1>                                                \
  > builtin_name(NonScalar1 x, NonScalar2 y, NonScalar3 z) noexcept {                         \
    detail::builtin_input_return_t<NonScalar1> ret;                                           \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {         \
      ret[i] = builtin_func(x[i], y[i], z[i]);                                                \
    }                                                                                         \
    return ret;                                                                               \
  }

#define HIPSYCL_BUILTIN_OVERLOADED_IMPL_MATH(h, template_h, builtin_name, builtin_func)    \
  h(float, builtin_name, builtin_func)                                                     \
  h(double, builtin_name, builtin_func)                                                    \
  h(half, builtin_name, detail::HalfAdapter{[](auto... a) { return builtin_func(a...); }}) \
  template_h(detail::is_valid_math_nonscalar, builtin_name, builtin_func)

#define HIPSYCL_BUILTIN_OVERLOADED_IMPL_NATIVE_MATH(h, template_h, builtin_name, builtin_func) \
  h(float, builtin_name, builtin_func)                                                         \
  template_h(detail::is_valid_special_math_nonscalar, builtin_name, builtin_func)

#define HIPSYCL_BUILTIN_OVERLOADED_IMPL_HALF_MATH(h, template_h, builtin_name, builtin_func) \
  h(float, builtin_name, builtin_func)                                                       \
  template_h(detail::is_valid_special_math_nonscalar, builtin_name, builtin_func)

#define HIPSYCL_MATH_BUILTIN(arity, builtin_name)                            \
  HIPSYCL_BUILTIN_OVERLOADED_IMPL_MATH(                                      \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_, arity),          \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_, arity), \
    builtin_name,                                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_, builtin_name)                    \
  )

#define HIPSYCL_NATIVE_MATH_BUILTIN(arity, builtin_name)                     \
  HIPSYCL_BUILTIN_OVERLOADED_IMPL_NATIVE_MATH(                               \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_, arity),          \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_, arity), \
    builtin_name,                                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_native_, builtin_name)             \
  )

#define HIPSYCL_HALF_MATH_BUILTIN(arity, builtin_name)                       \
  HIPSYCL_BUILTIN_OVERLOADED_IMPL_HALF_MATH(                                 \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_, arity),          \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_MATH_BUILTIN_GENERATOR_TEMPLATE_, arity), \
    builtin_name,                                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_half_, builtin_name)               \
  )

HIPSYCL_MATH_BUILTIN(UNARY_T, acos)
HIPSYCL_MATH_BUILTIN(UNARY_T, acosh)
HIPSYCL_MATH_BUILTIN(UNARY_T, acospi)
HIPSYCL_MATH_BUILTIN(UNARY_T, asin)
HIPSYCL_MATH_BUILTIN(UNARY_T, asinh)
HIPSYCL_MATH_BUILTIN(UNARY_T, asinpi)
HIPSYCL_MATH_BUILTIN(UNARY_T, atan)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, atan2)
HIPSYCL_MATH_BUILTIN(UNARY_T, atanh)
HIPSYCL_MATH_BUILTIN(UNARY_T, atanpi)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, atan2pi)
HIPSYCL_MATH_BUILTIN(UNARY_T, cbrt)
HIPSYCL_MATH_BUILTIN(UNARY_T, ceil)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, copysign)
HIPSYCL_MATH_BUILTIN(UNARY_T, cos)
HIPSYCL_MATH_BUILTIN(UNARY_T, cosh)
HIPSYCL_MATH_BUILTIN(UNARY_T, cospi)
HIPSYCL_MATH_BUILTIN(UNARY_T, erfc)
HIPSYCL_MATH_BUILTIN(UNARY_T, erf)
HIPSYCL_MATH_BUILTIN(UNARY_T, exp)
HIPSYCL_MATH_BUILTIN(UNARY_T, exp2)
HIPSYCL_MATH_BUILTIN(UNARY_T, exp10)
HIPSYCL_MATH_BUILTIN(UNARY_T, expm1)
HIPSYCL_MATH_BUILTIN(UNARY_T, fabs)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, fdim)
HIPSYCL_MATH_BUILTIN(UNARY_T, floor)
HIPSYCL_MATH_BUILTIN(TERNARY_T, fma)
HIPSYCL_MATH_BUILTIN(BINARY_T_TSCALAR, fmax)
HIPSYCL_MATH_BUILTIN(BINARY_T_TSCALAR, fmin)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, fmod)
HIPSYCL_MATH_BUILTIN(BINARY_T_TPTR, fract)
HIPSYCL_MATH_BUILTIN(BINARY_T_INTPTR, frexp)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, hypot)
HIPSYCL_MATH_BUILTIN(UNARY_T_RET_INT, ilogb)
HIPSYCL_MATH_BUILTIN(BINARY_T_INT_SCALAR, ldexp)
HIPSYCL_MATH_BUILTIN(UNARY_T, lgamma)
HIPSYCL_MATH_BUILTIN(BINARY_T_INTPTR, lgamma_r)
HIPSYCL_MATH_BUILTIN(UNARY_T, log)
HIPSYCL_MATH_BUILTIN(UNARY_T, log2)
HIPSYCL_MATH_BUILTIN(UNARY_T, log10)
HIPSYCL_MATH_BUILTIN(UNARY_T, log1p)
HIPSYCL_MATH_BUILTIN(UNARY_T, logb)
HIPSYCL_MATH_BUILTIN(TERNARY_T, mad)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, maxmag)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, minmag)
HIPSYCL_MATH_BUILTIN(BINARY_T_TPTR, modf)
// TODO nancode
HIPSYCL_MATH_BUILTIN(BINARY_T_T, nextafter)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, pow)
HIPSYCL_MATH_BUILTIN(BINARY_T_INT, pown)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, powr)
HIPSYCL_MATH_BUILTIN(BINARY_T_T, remainder)
// TODO remquo
HIPSYCL_MATH_BUILTIN(UNARY_T, rint)
HIPSYCL_MATH_BUILTIN(BINARY_T_INT, rootn)
HIPSYCL_MATH_BUILTIN(UNARY_T, round)
HIPSYCL_MATH_BUILTIN(UNARY_T, rsqrt)
HIPSYCL_MATH_BUILTIN(UNARY_T, sin)
HIPSYCL_MATH_BUILTIN(BINARY_T_TPTR, sincos)
HIPSYCL_MATH_BUILTIN(UNARY_T, sinh)
HIPSYCL_MATH_BUILTIN(UNARY_T, sinpi)
HIPSYCL_MATH_BUILTIN(UNARY_T, sqrt)
HIPSYCL_MATH_BUILTIN(UNARY_T, tan)
HIPSYCL_MATH_BUILTIN(UNARY_T, tanh)
// TODO tanpi
HIPSYCL_MATH_BUILTIN(UNARY_T, tgamma)
HIPSYCL_MATH_BUILTIN(UNARY_T, trunc)

namespace native {
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, cos)
HIPSYCL_NATIVE_MATH_BUILTIN(BINARY_T_T, divide)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, exp)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, exp2)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, exp10)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, log)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, log2)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, log10)
HIPSYCL_NATIVE_MATH_BUILTIN(BINARY_T_T, powr)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, recip)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, rsqrt)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, sin)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, sqrt)
HIPSYCL_NATIVE_MATH_BUILTIN(UNARY_T, tan)
}

namespace half_precision {
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, cos)
HIPSYCL_HALF_MATH_BUILTIN(BINARY_T_T, divide)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, exp)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, exp2)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, exp10)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, log)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, log2)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, log10)
HIPSYCL_HALF_MATH_BUILTIN(BINARY_T_T, powr)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, recip)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, rsqrt)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, sin)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, sqrt)
HIPSYCL_HALF_MATH_BUILTIN(UNARY_T, tan)
}
}

#endif
