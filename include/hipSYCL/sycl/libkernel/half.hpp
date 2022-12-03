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
public:
  half() = default;
  
  half(float f) noexcept
  : _data{f} {}
  
  half(double f) noexcept
  : _data{f} {}

  half(fp16::generic_half f) noexcept
  : _data{f} {}

  half(const half&) = default;

  half& operator=(const half&) noexcept = default;

  operator float() const {
    return _data.promote_to_float();
  }

  operator double() const {
    return _data.promote_to_double();
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator+(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::generic_half::builtin_add(a._data, b._data),
      return __hipsycl_sscp_half_add(a._data, b._data),
      return __hadd(a.cuda_representation, b.cuda_representation),
      // HIP uses compiler builtin addition for native _Float16 type
      return fp16::generic_half::builtin_add(a._data, b._data),
      return fp16::generic_half::builtin_add(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator-(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::generic_half::builtin_sub(a._data, b._data),
      return __hipsycl_sscp_half_sub(a._data, b._data),
      return __hsub(a.cuda_representation, b.cuda_representation),
      // HIP uses compiler builtin subtraction for native _Float16 type
      return fp16::generic_half::builtin_sub(a._data, b._data),
      return fp16::generic_half::builtin_sub(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator*(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::generic_half::builtin_mul(a._data, b._data),
      return __hipsycl_sscp_half_mul(a._data, b._data),
      return __hmul(a.cuda_representation, b.cuda_representation),
      // HIP uses compiler builtin mul for native _Float16 type
      return fp16::generic_half::builtin_sub(a._data, b._data),
      return fp16::generic_half::builtin_sub(a._data, b._data))
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend half operator/(const half& a, const half& b) noexcept {
    __hipsycl_backend_switch(
      return fp16::generic_half::builtin_div(a._data, b._data),
      return __hipsycl_sscp_half_div(a._data, b._data),
      return __hdiv(a.cuda_representation, b.cuda_representation),
      // HIP uses compiler builtin div for native _Float16 type
      return fp16::generic_half::builtin_div(a._data, b._data),
      return fp16::generic_half::builtin_div(a._data, b._data))
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
private:
  fp16::generic_half _data;
};

}
}

#endif
