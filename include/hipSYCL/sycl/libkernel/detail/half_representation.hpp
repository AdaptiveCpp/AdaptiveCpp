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
#ifndef HIPSYCL_SYCL_HALF_REPRESENTATION_HPP
#define HIPSYCL_SYCL_HALF_REPRESENTATION_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/bit_cast.hpp"
#include "int_types.hpp"

#ifdef __clang__
// It seems at least some versions of clang on x86 do not automatically
// link against the correct compiler-rt builtin library to allow
// for __fp16 to float conversion. Disable for now until we understand
// when we can actually enable __fp16.

#if (defined(__x86_64__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||           \
    (defined(__arm__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||              \
    (defined(__aarch64__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||          \
    ((defined(__AMDGPU__) || defined(__SPIR__) || defined(__SPIR64__)) &&      \
     (ACPP_LIBKERNEL_IS_DEVICE_PASS ||                                      \
      defined(HIPSYCL_SSCP_LIBKERNEL_LIBRARY)))
// These targets support _Float16
#define HIPSYCL_HALF_HAS_FLOAT16_TYPE
#endif

#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
  #define HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
#endif

#include "fp16/fp16.h"


namespace hipsycl::fp16 {

// Do not use a class to ensure consistent ABI between SSCP bitcode libraries
// and client code, which will have different LLVM target triple initially.
using half_storage = __acpp_uint16;

namespace detail {
// Currently we cannot call sycl::bit_cast directly, since we do not
// have ACPP_UNIVERSAL_TARGET attributes available here, which
// are needed for sycl::bit_cast.
template<class Tout, class Tin>
Tout bit_cast(Tin x) {
  Tout result;
  HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, x, result);
  return result;
}


#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
inline __acpp_uint16 native_float16_to_int(_Float16 x) noexcept {
  return bit_cast<__acpp_uint16>(x);
}

static _Float16 int_to_native_float16(__acpp_uint16 x) noexcept {
  return bit_cast<_Float16>(x);
}
#endif

#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
inline __acpp_uint16 cuda_half_to_int(__half x) noexcept {
  return bit_cast<__acpp_uint16>(x);
}

inline __half int_to_cuda_half(__acpp_uint16 x) noexcept {
  return bit_cast<__half>(x);
}
#endif

}



inline half_storage truncate_from(float f) noexcept {
  return hipsycl::fp16::fp16_ieee_from_fp32_value(f);
}

inline half_storage truncate_from(double f) noexcept {
  return truncate_from(static_cast<float>(f));
}

inline float promote_to_float(half_storage h) noexcept {
  return hipsycl::fp16::fp16_ieee_to_fp32_value(h);
}

inline double promote_to_double(half_storage h) noexcept {
  return static_cast<double>(promote_to_float(h));
}

inline half_storage create(float f) noexcept {
  return truncate_from(f);
}

inline half_storage create(double f) noexcept {
  return truncate_from(f);
}

inline half_storage create(__acpp_uint16 i) noexcept {
  return i;
}

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
inline half_storage create(_Float16 f) noexcept {
  return detail::native_float16_to_int(f);
}

inline _Float16 as_native_float16(half_storage h) noexcept {
  return detail::int_to_native_float16(h);
}
#endif

#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE

inline half_storage create(__half h) noexcept {
  return detail::cuda_half_to_int(h);
}

inline __half as_cuda_half(half_storage h) noexcept {
  return detail::int_to_cuda_half(h);
}
#endif

inline __acpp_uint16 as_integer(half_storage h) noexcept {
  return h;
}

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
using native_t = _Float16;
#else
using native_t = unsigned short;
#endif

// Provide basic "builtin" arithmetic functions that rely on only the compiler.
inline half_storage builtin_add(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return create(as_native_float16(a) + as_native_float16(b));
#else
  return create(promote_to_float(a) + promote_to_float(b));
#endif
}

inline half_storage builtin_sub(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return create(as_native_float16(a) - as_native_float16(b));
#else
  return create(promote_to_float(a) - promote_to_float(b));
#endif
}

inline half_storage builtin_mul(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return create(as_native_float16(a) * as_native_float16(b));
#else
  return create(promote_to_float(a) * promote_to_float(b));
#endif
}

inline half_storage builtin_div(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return create(as_native_float16(a) / as_native_float16(b));
#else
  return create(promote_to_float(a) / promote_to_float(b));
#endif
}

inline bool builtin_less_than(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return as_native_float16(a) < as_native_float16(b);
#else
  return promote_to_float(a) < promote_to_float(b);
#endif
}

inline bool builtin_less_than_equal(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return as_native_float16(a) <= as_native_float16(b);
#else
  return promote_to_float(a) <= promote_to_float(b);
#endif
}

inline bool builtin_greater_than(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return as_native_float16(a) > as_native_float16(b);
#else
  return promote_to_float(a) > promote_to_float(b);
#endif
}

inline bool builtin_greater_than_equal(half_storage a,
                                       half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  return as_native_float16(a) >= as_native_float16(b);
#else
  return promote_to_float(a) >= promote_to_float(b);
#endif
}
}

#endif
