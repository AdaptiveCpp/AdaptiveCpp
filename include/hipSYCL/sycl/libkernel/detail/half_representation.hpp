/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_SYCL_HALF_REPRESENTATION_HPP
#define HIPSYCL_SYCL_HALF_REPRESENTATION_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "int_types.hpp"

#ifdef __clang__
// It seems at least some versions of clang on x86 do not automatically
// link against the correct compiler-rt builtin library to allow
// for __fp16 to float conversion. Disable for now until we understand
// when we can actually enable __fp16.

#if (defined(__x86_64__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||           \
    (defined(__arm__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||              \
    (defined(__aarch64__) && defined(HIPSYCL_ENABLE_HALF_ON_HOST)) ||          \
    (defined(__AMDGPU__) || defined(__SPIR__) || defined(__SPIR64__))
  // These targets support _Float16
  #define HIPSYCL_HALF_HAS_FLOAT16_TYPE
#endif

#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
  #define HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
#endif

#include "fp16/fp16.h"


namespace hipsycl::fp16 {

struct generic_half {
  union {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    _Float16 native_fp16_representation;
#endif
#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
    __half cuda_representation;
#endif
    __hipsycl_uint16 int_representation;
  };

  generic_half() = default;
  generic_half(float f) {
    truncate_from(f);
  }
  generic_half(double f) {
    truncate_from(f);
  }

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  generic_half(_Float16 f)
  : native_fp16_representation{f} {}
#endif

#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
  generic_half(__half f)
  : cuda_representation{f} {}
#endif

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  using native_t = _Float16;
#else
  using native_t = unsigned short;
#endif

  void truncate_from(float f) noexcept {
    int_representation =
        hipsycl::fp16::fp16_ieee_from_fp32_value(f);
  }

  void truncate_from(double f) noexcept {
    truncate_from(static_cast<float>(f));
  }


  float promote_to_float() const noexcept {
    return hipsycl::fp16::fp16_ieee_to_fp32_value(
        int_representation);
  }

  double promote_to_double() const noexcept {
    return static_cast<double>(promote_to_float());
  }

  static generic_half builtin_add(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return generic_half{a.native_fp16_representation +
                                         b.native_fp16_representation};
#else
    return generic_half{a.promote_to_float() +
                                         b.promote_to_float()};
#endif
  }

  static generic_half builtin_sub(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return generic_half{a.native_fp16_representation -
                                         b.native_fp16_representation};
#else
    return generic_half{a.promote_to_float() -
                                         b.promote_to_float()};
#endif
  }

  static generic_half builtin_mul(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return generic_half{a.native_fp16_representation *
                                         b.native_fp16_representation};
#else
    return generic_half{a.promote_to_float() *
                                         b.promote_to_float()};
#endif
  }

  static generic_half builtin_div(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return generic_half{a.native_fp16_representation /
                                         b.native_fp16_representation};
#else
    return generic_half{a.promote_to_float() /
                                         b.promote_to_float()};
#endif
  }

  static bool builtin_less_than(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.native_fp16_representation < b.native_fp16_representation;
#else
    return a.promote_to_float() < b.promote_to_float();
#endif
  }

  static bool builtin_less_than_equal(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.native_fp16_representation <= b.native_fp16_representation;
#else
    return a.promote_to_float() <= b.promote_to_float();
#endif
  }

  static bool builtin_greater_than(generic_half a, generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.native_fp16_representation > b.native_fp16_representation;
#else
    return a.promote_to_float() > b.promote_to_float();
#endif
  }

  static bool builtin_greater_than_equal(generic_half a,
                                         generic_half b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.native_fp16_representation >= b.native_fp16_representation;
#else
    return a.promote_to_float() >= b.promote_to_float();
#endif
  }
};

}

#endif