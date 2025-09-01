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
#ifndef HIPSYCL_RELATIONAL_BUILTINS_HPP
#define HIPSYCL_RELATIONAL_BUILTINS_HPP

#include <cstdint>
#include "builtin_utils.hpp"

#include "hipSYCL/sycl/libkernel/builtin_interface.hpp"

namespace hipsycl::sycl {
namespace detail {
  template <typename T>
  struct is_valid_relational_scalar : is_in_type_set<float, double, half>::scalar_tester<T> {};

  template <typename T>
  struct is_valid_relational_nonscalar : std::bool_constant<
    is_valid_relational_scalar<builtin_input_element_t<T>>::value &&
    builtin_input_is_nonscalar_v<T>
  > {};
}

#define HIPSYCL_RELATIONAL_BUILTIN_GENERATOR_UNARY_T_RET_BOOL(T, builtin_name, builtin_func) \
  HIPSYCL_BUILTIN bool builtin_name(T x) noexcept {                                          \
    return builtin_func(x);                                                                  \
  }

#define HIPSYCL_RELATIONAL_BUILTIN_GENERATOR_TEMPLATE_UNARY_T_RET_BOOL(tester, builtin_name, builtin_func) \
  template<typename NonScalar1>                                                                            \
  HIPSYCL_BUILTIN                                                                                          \
  std::enable_if_t<                                                                                        \
    detail::builtin_inputs_are_compatible_v<                                                               \
      NonScalar1                                                                                           \
    > &&                                                                                                   \
    tester<NonScalar1>::value,                                                                             \
    detail::builtin_input_boollike_t<NonScalar1>                                                           \
  > builtin_name(NonScalar1 x) noexcept {                                                                  \
    detail::builtin_input_boollike_t<NonScalar1> ret;                                                      \
    for (std::size_t i = 0; i < detail::builtin_input_num_elems_v<NonScalar1>; i++) {                      \
      ret[i] = builtin_func(x[i]);                                                                         \
    }                                                                                                      \
    return ret;                                                                                            \
  }

#define HIPSYCL_RELATIONAL_BUILTIN_OVERLOADED_IMPL(h, template_h, builtin_name, builtin_func) \
  h(float, builtin_name, builtin_func)                                                        \
  h(double, builtin_name, builtin_func)                                                       \
  h(half, builtin_name, detail::HalfAdapter{[](auto... a) { return builtin_func(a...); }})    \
  template_h(detail::is_valid_relational_nonscalar, builtin_name, builtin_func)

#define HIPSYCL_RELATIONAL_BUILTIN(arity, builtin_name)                            \
  HIPSYCL_RELATIONAL_BUILTIN_OVERLOADED_IMPL(                                      \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_RELATIONAL_BUILTIN_GENERATOR_, arity),          \
    HIPSYCL_PP_CONCATENATE(HIPSYCL_RELATIONAL_BUILTIN_GENERATOR_TEMPLATE_, arity), \
    builtin_name,                                                                  \
    HIPSYCL_PP_CONCATENATE(detail::__acpp_, builtin_name)                          \
  )

HIPSYCL_RELATIONAL_BUILTIN(UNARY_T_RET_BOOL, isnan)
HIPSYCL_RELATIONAL_BUILTIN(UNARY_T_RET_BOOL, isinf)
HIPSYCL_RELATIONAL_BUILTIN(UNARY_T_RET_BOOL, isfinite)
HIPSYCL_RELATIONAL_BUILTIN(UNARY_T_RET_BOOL, isnormal)
HIPSYCL_RELATIONAL_BUILTIN(UNARY_T_RET_BOOL, signbit)
}
#endif
