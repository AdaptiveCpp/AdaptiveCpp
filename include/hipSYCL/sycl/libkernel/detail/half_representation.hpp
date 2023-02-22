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
#include "hipSYCL/sycl/libkernel/bit_cast.hpp"
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
     (HIPSYCL_LIBKERNEL_IS_DEVICE_PASS ||                                      \
      defined(HIPSYCL_SSCP_LIBKERNEL_LIBRARY)))
// These targets support _Float16
#define HIPSYCL_HALF_HAS_FLOAT16_TYPE
#endif

#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
  #define HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
#endif

#include "fp16/fp16.h"


namespace hipsycl::fp16 {

struct half_storage {
private:
  __hipsycl_uint16 _val;

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  static __hipsycl_uint16 native_float16_to_int(_Float16 x) noexcept {
    return sycl::bit_cast<__hipsycl_uint16>(x);
  }

  static _Float16 int_to_native_float16(__hipsycl_uint16 x) noexcept {
    return sycl::bit_cast<_Float16>(x);
  }
#endif

#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
  static __hipsycl_uint16 cuda_half_to_int(__half x) noexcept {
    return sycl::bit_cast<__hipsycl_uint16>(x);
  }

  static __half int_to_cuda_half(__hipsycl_uint16 x) noexcept {
    return sycl::bit_cast<__half>(x);
  }
#endif
public:

  half_storage() = default;
  half_storage(float f) {
    truncate_from(f);
  }
  half_storage(double f) {
    truncate_from(f);
  }

  explicit half_storage(__hipsycl_uint16 i)
  : _val{i} {}

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  half_storage(_Float16 f)
  : _val{native_float16_to_int(f)} {}

  half_storage& operator=(_Float16 f) noexcept{
    _val = native_float16_to_int(f);
    return *this;
  }

  _Float16 as_native_float16() const noexcept {
    return int_to_native_float16(_val);
  }
#endif

#ifdef HIPSYCL_HALF_HAS_CUDA_HALF_TYPE
  half_storage(__half f)
  : _val{cuda_half_to_int(f)} {}

  half_storage& operator=(__half f) noexcept{
    _val = cuda_half_to_int(f);
    return *this;
  }

  __half as_cuda_half() const noexcept {
    return int_to_cuda_half(_val);
  }
#endif

  __hipsycl_uint16 as_integer() const noexcept {
    return _val;
  }

#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
  using native_t = _Float16;
#else
  using native_t = unsigned short;
#endif


  void truncate_from(float f) noexcept {
    _val = hipsycl::fp16::fp16_ieee_from_fp32_value(f);
  }

  void truncate_from(double f) noexcept {
    truncate_from(static_cast<float>(f));
  }


  float promote_to_float() const noexcept {
    return hipsycl::fp16::fp16_ieee_to_fp32_value(_val);
  }

  double promote_to_double() const noexcept {
    return static_cast<double>(promote_to_float());
  }

  // Provide basic "builtin" arithmetic functions that rely on only the compiler.
  static half_storage builtin_add(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return half_storage{a.as_native_float16() + b.as_native_float16()};
#else
    return half_storage{a.promote_to_float() + b.promote_to_float()};
#endif
  }

  static half_storage builtin_sub(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return half_storage{a.as_native_float16() - b.as_native_float16()};
#else
    return half_storage{a.promote_to_float() - b.promote_to_float()};
#endif
  }

  static half_storage builtin_mul(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return half_storage{a.as_native_float16() * b.as_native_float16()};
#else
    return half_storage{a.promote_to_float() * b.promote_to_float()};
#endif
  }

  static half_storage builtin_div(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return half_storage{a.as_native_float16() / b.as_native_float16()};
#else
    return half_storage{a.promote_to_float() / b.promote_to_float()};
#endif
  }

  static bool builtin_less_than(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.as_native_float16() < b.as_native_float16();
#else
    return a.promote_to_float() < b.promote_to_float();
#endif
  }

  static bool builtin_less_than_equal(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.as_native_float16() <= b.as_native_float16();
#else
    return a.promote_to_float() <= b.promote_to_float();
#endif
  }

  static bool builtin_greater_than(half_storage a, half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.as_native_float16() > b.as_native_float16();
#else
    return a.promote_to_float() > b.promote_to_float();
#endif
  }

  static bool builtin_greater_than_equal(half_storage a,
                                         half_storage b) noexcept {
#ifdef HIPSYCL_HALF_HAS_FLOAT16_TYPE
    return a.as_native_float16() >= b.as_native_float16();
#else
    return a.promote_to_float() >= b.promote_to_float();
#endif
  }
};

}

#endif
