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

#ifndef HIPSYCL_METAL_SCAN_HELPERS_HPP
#define HIPSYCL_METAL_SCAN_HELPERS_HPP

#include "helpers.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {
namespace detail::metal_builtins {


#define SCAN_TYPES \
  X(i8) \
  X(i16) \
  X(i32) \
  X(i64) \
  X(u8) \
  X(u16) \
  X(u32) \
  X(u64) \
  X(f16) \
  X(f32)

#define X(type) HIPSYCL_SSCP_BUILTIN type __acpp_sscp_metal_scan_##type(const char* s, type value);
SCAN_TYPES
#undef X

template <typename T>
inline T __acpp_sscp_metal_scan(const char* s, T value) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_scan_##type(s, value); \
  }

SCAN_TYPES
#undef X
}

template <__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_sub_group_inclusive_scan(T value) {
  if constexpr (op == __acpp_sscp_algorithm_op::plus) {
    return __acpp_sscp_metal_scan<T>("simd_prefix_inclusive_sum", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::multiply) {
    return __acpp_sscp_metal_scan<T>("simd_prefix_inclusive_product", value);
  } else {
    static_assert(op == __acpp_sscp_algorithm_op::plus || op == __acpp_sscp_algorithm_op::multiply, "Unsupported operation");
  }
}

template <__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_sub_group_exclusive_scan(T value) {
  if constexpr (op == __acpp_sscp_algorithm_op::plus) {
    return __acpp_sscp_metal_scan<T>("simd_prefix_exclusive_sum", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::multiply) {
    return __acpp_sscp_metal_scan<T>("simd_prefix_exclusive_product", value);
  } else {
    static_assert(op == __acpp_sscp_algorithm_op::plus || op == __acpp_sscp_algorithm_op::multiply, "Unsupported operation");
  }
}

}
}
}

#endif