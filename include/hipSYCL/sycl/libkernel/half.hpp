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

#ifndef HIPSYCL_HALF_HPP
#define HIPSYCL_HALF_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#endif


namespace hipsycl {
namespace sycl {

class half {
private:
  half(fp16::half_storage f) noexcept
  : _data{f} {}
public:
  half() = default;
  
  explicit half(float f) noexcept
  : _data{fp16::create(f)} {}
  
  explicit half(double f) noexcept
  : _data{fp16::create(f)} {}

  explicit half(int f) noexcept
  : _data{fp16::create(static_cast<float>(f))} {}


  half(const half&) = default;

  half& operator=(const half&) noexcept = default;

  operator float() const {
    return fp16::promote_to_float(_data);
  }

  operator double() const {
    return fp16::promote_to_double(_data);
  }

  operator int() const {
    return static_cast<int>(fp16::promote_to_float(_data));
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator+(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_add(a._data, b._data),
      return __hipsycl_sscp_half_add(a._data, b._data),
      return fp16::create(__hadd(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin addition for native _Float16 type
      return fp16::builtin_add(a._data, b._data),
      return fp16::builtin_add(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator-(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_sub(a._data, b._data),
      return __hipsycl_sscp_half_sub(a._data, b._data),
      return fp16::create(__hsub(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin subtraction for native _Float16 type
      return fp16::builtin_sub(a._data, b._data),
      return fp16::builtin_sub(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator*(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_mul(a._data, b._data),
      return __hipsycl_sscp_half_mul(a._data, b._data),
      return fp16::create(__hmul(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin mul for native _Float16 type
      return fp16::builtin_mul(a._data, b._data),
      return fp16::builtin_mul(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator/(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_div(a._data, b._data),
      return __hipsycl_sscp_half_div(a._data, b._data),
      return fp16::create(__hdiv(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin div for native _Float16 type
      return fp16::builtin_div(a._data, b._data),
      return fp16::builtin_div(a._data, b._data))
  }

  friend half& operator+=(half& a, const half& b) noexcept {
    a = a + b;
    return a;
  }

  friend half& operator-=(half& a, const half& b) noexcept {
    a = a - b;
    return a;
  }

  friend half& operator*=(half& a, const half& b) noexcept {
    a = a * b;
    return a;
  }

  friend half& operator/=(half& a, const half& b) noexcept {
    a = a / b;
    return a;
  }

  friend bool operator==(const half& a, const half& b) noexcept {
    return a._data == b._data;
  }

  friend bool operator!=(const half& a, const half& b) noexcept {
    return a._data != b._data;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_less_than(a._data, b._data),
      return __hipsycl_sscp_half_lt(a._data, b._data),
      return __hlt(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_less_than(a._data, b._data),
      return fp16::builtin_less_than(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator<=(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_less_than_equal(a._data, b._data),
      return __hipsycl_sscp_half_lte(a._data, b._data),
      return __hle(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_less_than_equal(a._data, b._data),
      return fp16::builtin_less_than_equal(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_greater_than(a._data, b._data),
      return __hipsycl_sscp_half_gt(a._data, b._data),
      return __hgt(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_greater_than(a._data, b._data),
      return fp16::builtin_greater_than(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend bool operator>=(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::builtin_greater_than_equal(a._data, b._data),
      return __hipsycl_sscp_half_gte(a._data, b._data),
      return __hge(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_greater_than_equal(a._data, b._data),
      return fp16::builtin_greater_than_equal(a._data, b._data))
  }
private:
  fp16::half_storage _data;
};

}
}

#endif
