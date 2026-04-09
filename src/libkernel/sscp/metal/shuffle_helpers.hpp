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

#ifndef HIPSYCL_METAL_SHUFFLE_HELPERS_HPP
#define HIPSYCL_METAL_SHUFFLE_HELPERS_HPP

#include "helpers.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {
namespace detail::metal_builtins {

#define SHUFFLE_TYPES \
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

#define X(type) HIPSYCL_SSCP_BUILTIN type __acpp_sscp_metal_shuffle_##type(const char* s, type value, u32 offset);
SHUFFLE_TYPES
#undef X

template <typename T>
inline T __acpp_sscp_metal_shuffle_down(T value, u32 delta) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_shuffle_##type("simd_shuffle_down", value, delta); \
  }

SHUFFLE_TYPES

#undef X
}

template <typename T>
inline T __acpp_sscp_metal_shuffle_up(T value, u32 delta) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_shuffle_##type("simd_shuffle_up", value, delta); \
  }

SHUFFLE_TYPES

#undef X
}

template <typename T>
inline T __acpp_sscp_metal_shuffle_xor(T value, u32 mask) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_shuffle_##type("simd_shuffle_xor", value, mask); \
  }

SHUFFLE_TYPES

#undef X
}

template <typename T>
inline T __acpp_sscp_metal_shuffle(T value, u32 lane) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_shuffle_##type("simd_shuffle", value, lane); \
  }

SHUFFLE_TYPES

#undef X
}

}
}
}

#endif