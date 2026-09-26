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
#include "shuffle_helpers.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/scan_subgroup.hpp"

#include <limits>
#include <type_traits>

namespace hipsycl {
namespace sycl {
namespace detail::metal_builtins {


#define SCAN_INT_TYPES \
  X(i8) \
  X(i16) \
  X(i32) \
  X(i64) \
  X(u8) \
  X(u16) \
  X(u32) \
  X(u64)

#define SCAN_FLOAT_TYPES \
  X(f16) \
  X(f32)

#define SCAN_TYPES \
  SCAN_INT_TYPES \
  SCAN_FLOAT_TYPES

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

// MSL has no 64-bit prefix sum
// sum 16-bit chunks in 32 bits then add them together to propagate carries
template <typename T>
inline T __acpp_sscp_metal_inclusive_scan_sum_64(T value) {
  const u64 x = (u64)value;

  u32 x0 = (u32)x & 0xffff;
  u32 x1 = (u32)(x >> 16) & 0xffff;
  u32 x2 = (u32)(x >> 32) & 0xffff;
  u32 x3 = (u32)(x >> 48) & 0xffff;

  u32 p0 = __acpp_sscp_metal_scan<u32>("simd_prefix_inclusive_sum", x0);
  u32 p1 = __acpp_sscp_metal_scan<u32>("simd_prefix_inclusive_sum", x1);
  u32 p2 = __acpp_sscp_metal_scan<u32>("simd_prefix_inclusive_sum", x2);
  u32 p3 = __acpp_sscp_metal_scan<u32>("simd_prefix_inclusive_sum", x3);

  u64 prefix = ((u64)p3 << 48) + ((u64)p2 << 32) + ((u64)p1 << 16) + (u64)p0;
  return (T)prefix;
}

template <typename T>
inline T __acpp_sscp_metal_exclusive_scan_sum_64(T value) {
  const u64 x = (u64)value;

  u32 x0 = (u32)x & 0xffff;
  u32 x1 = (u32)(x >> 16) & 0xffff;
  u32 x2 = (u32)(x >> 32) & 0xffff;
  u32 x3 = (u32)(x >> 48) & 0xffff;

  u32 p0 = __acpp_sscp_metal_scan<u32>("simd_prefix_exclusive_sum", x0);
  u32 p1 = __acpp_sscp_metal_scan<u32>("simd_prefix_exclusive_sum", x1);
  u32 p2 = __acpp_sscp_metal_scan<u32>("simd_prefix_exclusive_sum", x2);
  u32 p3 = __acpp_sscp_metal_scan<u32>("simd_prefix_exclusive_sum", x3);

  u64 prefix = ((u64)p3 << 48) + ((u64)p2 << 32) + ((u64)p1 << 16) + (u64)p0;
  return (T)prefix;
}

template <__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_sub_group_inclusive_scan(T value) {
  if constexpr (op == __acpp_sscp_algorithm_op::plus) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
      return __acpp_sscp_metal_inclusive_scan_sum_64<T>(value);
    } else {
      return __acpp_sscp_metal_scan<T>("simd_prefix_inclusive_sum", value);
    }
  } else if constexpr (op == __acpp_sscp_algorithm_op::multiply) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
      using BinaryOp = typename hipsycl::libkernel::sscp::get_op<op>::type;
      return hipsycl::libkernel::sscp::sg_inclusive_scan(value, BinaryOp{});
    } else {
      return __acpp_sscp_metal_scan<T>("simd_prefix_inclusive_product", value);
    }
  } else if constexpr (
    op == __acpp_sscp_algorithm_op::min ||
    op == __acpp_sscp_algorithm_op::max ||
    op == __acpp_sscp_algorithm_op::bit_and ||
    op == __acpp_sscp_algorithm_op::bit_or ||
    op == __acpp_sscp_algorithm_op::bit_xor)
  {
    static_assert(std::is_integral_v<T> || (op == __acpp_sscp_algorithm_op::min || op == __acpp_sscp_algorithm_op::max));
    using BinaryOp = typename hipsycl::libkernel::sscp::get_op<op>::type;
    return hipsycl::libkernel::sscp::sg_inclusive_scan(value, BinaryOp{});
  } else {
    static_assert(op == __acpp_sscp_algorithm_op::plus, "Unsupported scan operation");
  }
}

template <__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_sub_group_exclusive_scan(T value, T init) {
  if constexpr (op == __acpp_sscp_algorithm_op::plus) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
      return __acpp_sscp_metal_exclusive_scan_sum_64<T>(value);
    } else {
      return __acpp_sscp_metal_scan<T>("simd_prefix_exclusive_sum", value);
    }
  } else if constexpr (op == __acpp_sscp_algorithm_op::multiply) {
    if constexpr (std::is_integral_v<T> && sizeof(T) == 8) {
      using BinaryOp = typename hipsycl::libkernel::sscp::get_op<op>::type;
      return hipsycl::libkernel::sscp::sg_exclusive_scan(value, BinaryOp{}, init);
    } else {
      return __acpp_sscp_metal_scan<T>("simd_prefix_exclusive_product", value);
    }
  } else if constexpr (
    op == __acpp_sscp_algorithm_op::min ||
    op == __acpp_sscp_algorithm_op::max ||
    op == __acpp_sscp_algorithm_op::bit_and ||
    op == __acpp_sscp_algorithm_op::bit_or ||
    op == __acpp_sscp_algorithm_op::bit_xor)
  {
    static_assert(std::is_integral_v<T> || (op == __acpp_sscp_algorithm_op::min || op == __acpp_sscp_algorithm_op::max));
    using BinaryOp = typename hipsycl::libkernel::sscp::get_op<op>::type;
    return hipsycl::libkernel::sscp::sg_exclusive_scan(value, BinaryOp{}, init);
  } else {
    static_assert(op == __acpp_sscp_algorithm_op::plus, "Unsupported scan operation");
  }
}

}
}
}

#endif
