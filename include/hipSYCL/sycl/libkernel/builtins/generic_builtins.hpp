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
#ifndef HIPSYCL_INTEGER_BUILTINS_HPP
#define HIPSYCL_INTEGER_BUILTINS_HPP

#include <cstdint>
#include "builtin_utils.hpp"

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/builtin_interface.hpp"

namespace hipsycl::sycl {
namespace detail {
  template <typename T>
  struct is_valid_generic_integer_scalar : is_in_type_set<
    char, signed char, short, int, long, long long,
    unsigned char, unsigned short, unsigned int, unsigned long, unsigned long long
  >::scalar_tester<T> {};

  // For ease of implementation, ensure the fixed width integer types
  // are aliased to the standard integer types.
  static_assert(is_valid_generic_integer_scalar<int8_t>::value);
  static_assert(is_valid_generic_integer_scalar<int16_t>::value);
  static_assert(is_valid_generic_integer_scalar<int32_t>::value);
  static_assert(is_valid_generic_integer_scalar<int64_t>::value);
  static_assert(is_valid_generic_integer_scalar<uint8_t>::value);
  static_assert(is_valid_generic_integer_scalar<uint16_t>::value);
  static_assert(is_valid_generic_integer_scalar<uint32_t>::value);
  static_assert(is_valid_generic_integer_scalar<uint64_t>::value);

  template <typename T>
  struct is_valid_generic_float_scalar : is_in_type_set<float, double, half>::scalar_tester<T> {};

  template <typename T>
  struct is_valid_generic_integer : std::bool_constant<
    is_valid_generic_integer_scalar<builtin_input_element_t<T>>::value
  > {};

  template <typename T>
  struct is_valid_generic_integer_32 : std::bool_constant<
    std::is_same_v<builtin_input_element_t<T>, uint32_t> || std::is_same_v<builtin_input_element_t<T>, int32_t>
  > {};

  template <typename T>
  struct is_valid_generic_float : std::bool_constant<
    is_valid_generic_float_scalar<builtin_input_element_t<T>>::value
  > {};

  template <typename T>
  struct is_valid_generic_float_geometric : std::bool_constant<
    is_valid_generic_float_scalar<builtin_input_element_t<T>>::value &&
    (
      !builtin_input_is_nonscalar_v<T> ||
      builtin_input_num_elems_v<T> == 2 ||
      builtin_input_num_elems_v<T> == 3 ||
      builtin_input_num_elems_v<T> == 4
    )
  > {};
}

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T(tester, builtin_name, builtin_func)     \
  template<typename Gen1>                                                         \
  HIPSYCL_BUILTIN                                                                 \
  std::enable_if_t<                                                               \
    detail::builtin_inputs_are_compatible_v<Gen1> &&                              \
    tester<Gen1>::value,                                                          \
    detail::builtin_input_return_t<Gen1>                                          \
  > builtin_name(Gen1 x) noexcept {                                               \
    if constexpr (!detail::builtin_input_is_nonscalar_v<Gen1>) {                  \
      return builtin_func(                                                        \
        detail::data_element(x, 0)                                                \
      );                                                                          \
    } else {                                                                      \
      detail::builtin_input_return_t<Gen1> ret;                                   \
      for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<Gen1>; i++) { \
        detail::data_element(ret, i) = builtin_func(                              \
          detail::data_element(x, i)                                              \
        );                                                                        \
      }                                                                           \
      return ret;                                                                 \
    }                                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T_REDUCTION(tester, builtin_name, builtin_func) \
  template<typename Gen1>                                                               \
  HIPSYCL_BUILTIN                                                                       \
  std::enable_if_t<                                                                     \
    detail::builtin_inputs_are_compatible_v<Gen1> &&                                    \
    tester<Gen1>::value,                                                                \
    detail::builtin_input_element_t<Gen1>                                               \
  > builtin_name(Gen1 x) noexcept {                                                     \
    return builtin_func(x);                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_UNARY_T_TRANSFORM(tester, builtin_name, builtin_func) \
  template<typename Gen1>                                                               \
  HIPSYCL_BUILTIN                                                                       \
  std::enable_if_t<                                                                     \
    detail::builtin_inputs_are_compatible_v<Gen1> &&                                    \
    tester<Gen1>::value,                                                                \
    detail::builtin_input_return_t<Gen1>                                               \
  > builtin_name(Gen1 x) noexcept {                                                     \
    return builtin_func(x);                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T(tester, builtin_name, builtin_func)  \
  template<typename Gen1, typename Gen2>                                          \
  HIPSYCL_BUILTIN                                                                 \
  std::enable_if_t<                                                               \
    detail::builtin_inputs_are_compatible_v<Gen1, Gen2> &&                        \
    tester<Gen1>::value,                                                          \
    detail::builtin_input_return_t<Gen1>                                          \
  > builtin_name(Gen1 x, Gen2 y) noexcept {                                       \
    if constexpr (!detail::builtin_input_is_nonscalar_v<Gen1>) {                  \
      return builtin_func(                                                        \
        detail::data_element(x, 0),                                               \
        detail::data_element(y, 0)                                                \
      );                                                                          \
    } else {                                                                      \
      detail::builtin_input_return_t<Gen1> ret;                                   \
      for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<Gen1>; i++) { \
        detail::data_element(ret, i) = builtin_func(                              \
          detail::data_element(x, i),                                             \
          detail::data_element(y, i)                                              \
        );                                                                        \
      }                                                                           \
      return ret;                                                                 \
    }                                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T_REDUCTION(tester, builtin_name, builtin_func) \
  template<typename Gen1, typename Gen2>                                                   \
  HIPSYCL_BUILTIN                                                                          \
  std::enable_if_t<                                                                        \
    detail::builtin_inputs_are_compatible_v<Gen1, Gen2> &&                                 \
    tester<Gen1>::value,                                                                   \
    detail::builtin_input_element_t<Gen1>                                                  \
  > builtin_name(Gen1 x, Gen2 y) noexcept {                                                \
    return builtin_func(x, y);                                                             \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_T_TSCALAR(tester, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T(tester, builtin_name, builtin_func)             \
  template<typename NonScalar1>                                                        \
  HIPSYCL_BUILTIN                                                                      \
  std::enable_if_t<                                                                    \
    detail::builtin_inputs_are_compatible_v<NonScalar1> &&                             \
    tester<NonScalar1>::value,                                                         \
    detail::builtin_input_return_t<NonScalar1>                                         \
  > builtin_name(                                                                      \
    NonScalar1 x,                                                                      \
    typename NonScalar1::value_type y                                                  \
  ) noexcept {                                                                         \
    detail::builtin_input_return_t<NonScalar1> ret;                                    \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {  \
      detail::data_element(ret, i) = builtin_func(                                     \
        detail::data_element(x, i),                                                    \
        y                                                                              \
      );                                                                               \
    }                                                                                  \
    return ret;                                                                        \
  }

#define HIPSYCL_BUILTIN_GENERATOR_BINARY_TSCALAR_T(tester, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN_GENERATOR_BINARY_T_T(tester, builtin_name, builtin_func)             \
  template<typename NonScalar1>                                                        \
  HIPSYCL_BUILTIN                                                                      \
  std::enable_if_t<                                                                    \
    detail::builtin_inputs_are_compatible_v<NonScalar1> &&                             \
    tester<NonScalar1>::value,                                                         \
    detail::builtin_input_return_t<NonScalar1>                                         \
  > builtin_name(                                                                      \
    typename NonScalar1::value_type x,                                                 \
    NonScalar1 y                                                                       \
  ) noexcept {                                                                         \
    detail::builtin_input_return_t<NonScalar1> ret;                                    \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {  \
      detail::data_element(ret, i) = builtin_func(                                     \
        x,                                                                             \
        detail::data_element(y, i)                                                     \
      );                                                                               \
    }                                                                                  \
    return ret;                                                                        \
  }

#define HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_T_T(tester, builtin_name, builtin_func) \
  template<typename Gen1, typename Gen2, typename Gen3>                             \
  HIPSYCL_BUILTIN                                                                   \
  std::enable_if_t<                                                                 \
    detail::builtin_inputs_are_compatible_v<Gen1, Gen2, Gen3> &&                    \
    tester<Gen1>::value,                                                            \
    detail::builtin_input_return_t<Gen1>                                            \
  > builtin_name(Gen1 x, Gen2 y, Gen3 z) noexcept {                                 \
    if constexpr (!detail::builtin_input_is_nonscalar_v<Gen1>) {                    \
      return builtin_func(                                                          \
        detail::data_element(x, 0),                                                 \
        detail::data_element(y, 0),                                                 \
        detail::data_element(z, 0)                                                  \
      );                                                                            \
    } else {                                                                        \
      detail::builtin_input_return_t<Gen1> ret;                                     \
      for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<Gen1>; i++) {   \
        detail::data_element(ret, i) = builtin_func(                                \
          detail::data_element(x, i),                                               \
          detail::data_element(y, i),                                               \
          detail::data_element(z, i)                                                \
        );                                                                          \
      }                                                                             \
      return ret;                                                                   \
    }                                                                               \
  }

#define HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_TSCALAR_TSCALAR(tester, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_T_T(tester, builtin_name, builtin_func)                   \
  template<typename NonScalar1>                                                                 \
  HIPSYCL_BUILTIN                                                                               \
  std::enable_if_t<                                                                             \
    detail::builtin_inputs_are_compatible_v<NonScalar1> &&                                      \
    tester<NonScalar1>::value,                                                                  \
    detail::builtin_input_return_t<NonScalar1>                                                  \
  > builtin_name(                                                                               \
    NonScalar1 x,                                                                               \
    typename NonScalar1::value_type y,                                                          \
    typename NonScalar1::value_type z                                                           \
  ) noexcept {                                                                                  \
    detail::builtin_input_return_t<NonScalar1> ret;                                             \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {           \
      detail::data_element(ret, i) = builtin_func(                                              \
        detail::data_element(x, i),                                                             \
        y,                                                                                      \
        z                                                                                       \
      );                                                                                        \
    }                                                                                           \
    return ret;                                                                                 \
  }

#define HIPSYCL_BUILTIN_GENERATOR_TERNARY_TSCALAR_TSCALAR_T(tester, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_T_T(tester, builtin_name, builtin_func)                   \
  template<typename NonScalar1>                                                                 \
  HIPSYCL_BUILTIN                                                                               \
  std::enable_if_t<                                                                             \
    detail::builtin_inputs_are_compatible_v<NonScalar1> &&                                      \
    tester<NonScalar1>::value,                                                                  \
    detail::builtin_input_return_t<NonScalar1>                                                  \
  > builtin_name(                                                                               \
    typename NonScalar1::value_type x,                                                          \
    typename NonScalar1::value_type y,                                                          \
    NonScalar1 z                                                                                \
  ) noexcept {                                                                                  \
    detail::builtin_input_return_t<NonScalar1> ret;                                             \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {           \
      detail::data_element(ret, i) = builtin_func(                                              \
        x,                                                                                      \
        y,                                                                                      \
        detail::data_element(z, i)                                                              \
      );                                                                                        \
    }                                                                                           \
    return ret;                                                                                 \
  }

#define HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_T_TSCALAR(tester, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN_GENERATOR_TERNARY_T_T_T(tester, builtin_name, builtin_func)             \
  template<typename NonScalar1, typename NonScalar2>                                      \
  HIPSYCL_BUILTIN                                                                         \
  std::enable_if_t<                                                                       \
    detail::builtin_inputs_are_compatible_v<NonScalar1, NonScalar2> &&                    \
    tester<NonScalar1>::value,                                                            \
    detail::builtin_input_return_t<NonScalar1>                                            \
  > builtin_name(                                                                         \
    NonScalar1 x,                                                                         \
    NonScalar2 y,                                                                         \
    typename NonScalar1::value_type z                                                     \
  ) noexcept {                                                                            \
    detail::builtin_input_return_t<NonScalar1> ret;                                       \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {     \
      detail::data_element(ret, i) = builtin_func(                                        \
        detail::data_element(x, i),                                                       \
        detail::data_element(y, i),                                                       \
        z                                                                                 \
      );                                                                                  \
    }                                                                                     \
    return ret;                                                                           \
  }

#define HIPSYCL_INTEGER_BUILTIN(arity, builtin_name)         \
  HIPSYCL_PP_CONCATENATE(HIPSYCL_BUILTIN_GENERATOR_, arity)( \
    detail::is_valid_generic_integer,                        \
    builtin_name,                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_, builtin_name)    \
  )

#define HIPSYCL_COMMON_BUILTIN(arity, builtin_name)          \
  HIPSYCL_PP_CONCATENATE(HIPSYCL_BUILTIN_GENERATOR_, arity)( \
    detail::is_valid_generic_float,                          \
    builtin_name,                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_, builtin_name)    \
  )

#define HIPSYCL_GEOMETRIC_BUILTIN(arity, builtin_name)       \
  HIPSYCL_PP_CONCATENATE(HIPSYCL_BUILTIN_GENERATOR_, arity)( \
    detail::is_valid_generic_float_geometric,                \
    builtin_name,                                            \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_, builtin_name)    \
  )

HIPSYCL_INTEGER_BUILTIN(UNARY_T, abs)
// TODO abs_diff
// TODO add_sat
// TODO hadd
// TODO rhadd
HIPSYCL_INTEGER_BUILTIN(TERNARY_T_TSCALAR_TSCALAR, clamp)
HIPSYCL_INTEGER_BUILTIN(UNARY_T, clz)
HIPSYCL_INTEGER_BUILTIN(UNARY_T, ctz)
// TODO mad_hi
// TODO mad_sat
HIPSYCL_INTEGER_BUILTIN(BINARY_T_TSCALAR, max)
HIPSYCL_INTEGER_BUILTIN(BINARY_T_TSCALAR, min)
// TODO mul_hi
// TODO rotate
// TODO sub_sat
// TODO upsample
HIPSYCL_INTEGER_BUILTIN(UNARY_T, popcount)
// TODO mad24
template <typename Gen1, typename Gen2>
std::enable_if_t<
  detail::builtin_inputs_are_compatible_v<Gen1, Gen2> &&
  detail::is_valid_generic_integer_32<Gen1>::value,
  detail::builtin_input_return_t<Gen1>
> mul24(Gen1 x, Gen2 y) noexcept {
  return detail::__acpp_mul24(x, y);
}


HIPSYCL_COMMON_BUILTIN(TERNARY_T_TSCALAR_TSCALAR, clamp)
HIPSYCL_COMMON_BUILTIN(UNARY_T, degrees)
HIPSYCL_COMMON_BUILTIN(BINARY_T_TSCALAR, max)
HIPSYCL_COMMON_BUILTIN(BINARY_T_TSCALAR, min)
HIPSYCL_COMMON_BUILTIN(TERNARY_T_T_TSCALAR, mix)
HIPSYCL_COMMON_BUILTIN(UNARY_T, radians)
HIPSYCL_COMMON_BUILTIN(BINARY_TSCALAR_T, step)
HIPSYCL_COMMON_BUILTIN(TERNARY_TSCALAR_TSCALAR_T, smoothstep)
HIPSYCL_COMMON_BUILTIN(UNARY_T, sign)


template<typename Gen1, typename Gen2>
HIPSYCL_BUILTIN
std::enable_if_t<
  detail::builtin_inputs_are_compatible_v<Gen1, Gen2> &&
  detail::is_valid_generic_float_geometric<Gen1>::value &&
  detail::builtin_input_num_elems_v<Gen1> == 3,
  detail::builtin_input_return_t<Gen1>
> cross(Gen1 x, Gen2 y) noexcept {
  return detail::__acpp_cross3(x, y);
}
template<typename Gen1, typename Gen2>
HIPSYCL_BUILTIN
std::enable_if_t<
  detail::builtin_inputs_are_compatible_v<Gen1, Gen2> &&
  detail::is_valid_generic_float_geometric<Gen1>::value &&
  detail::builtin_input_num_elems_v<Gen1> == 4,
  detail::builtin_input_return_t<Gen1>
> cross(Gen1 x, Gen2 y) noexcept {
  return detail::__acpp_cross4(x, y);
}
HIPSYCL_GEOMETRIC_BUILTIN(BINARY_T_T_REDUCTION, dot)
HIPSYCL_GEOMETRIC_BUILTIN(BINARY_T_T_REDUCTION, distance)
HIPSYCL_GEOMETRIC_BUILTIN(UNARY_T_REDUCTION, length)
HIPSYCL_GEOMETRIC_BUILTIN(UNARY_T_TRANSFORM, normalize)
HIPSYCL_GEOMETRIC_BUILTIN(BINARY_T_T_REDUCTION, fast_distance)
HIPSYCL_GEOMETRIC_BUILTIN(UNARY_T_REDUCTION, fast_length)
HIPSYCL_GEOMETRIC_BUILTIN(UNARY_T_TRANSFORM, fast_normalize)
}
#endif
