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
#ifndef HIPSYCL_HALF_HPP
#define HIPSYCL_HALF_HPP

#include <limits>
#include <functional>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#endif


namespace hipsycl {
namespace sycl {

class half;

namespace detail {
  constexpr half create_half(fp16::half_storage h);
  constexpr fp16::half_storage get_half_storage(half h);
}

class half {
private:
  friend constexpr half detail::create_half(fp16::half_storage h);
  friend constexpr fp16::half_storage detail::get_half_storage(half h);

public:
  constexpr half() : _data{} {};
  
  half(float f) noexcept
  : _data{fp16::create(f)} {}

  half(const half&) = default;

  half& operator=(const half&) noexcept = default;

  operator float() const {
    return fp16::promote_to_float(_data);
  }


  ACPP_UNIVERSAL_TARGET
  friend half operator+(const half& a, const half& b) noexcept {
    fp16::half_storage data;
    // __acpp_backend_switch contains an if statement for sscp pass, so we
    // cannot write `fp16::half_storage data = __acpp_backend_switch(...)`.
    __acpp_backend_switch(
      data = fp16::builtin_add(a._data, b._data),
      data = __acpp_sscp_half_add(a._data, b._data),
      data = fp16::create(__hadd(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin addition for native _Float16 type
      data = fp16::builtin_add(a._data, b._data));
    return detail::create_half(data);
  }

  ACPP_UNIVERSAL_TARGET
  friend half operator-(const half& a, const half& b) noexcept {
    fp16::half_storage data;
    __acpp_backend_switch(
      data = fp16::builtin_sub(a._data, b._data),
      data = __acpp_sscp_half_sub(a._data, b._data),
      data = fp16::create(__hsub(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin subtraction for native _Float16 type
      data = fp16::builtin_sub(a._data, b._data));
    return detail::create_half(data);
  }

  ACPP_UNIVERSAL_TARGET
  friend half operator*(const half& a, const half& b) noexcept {
    fp16::half_storage data;
    __acpp_backend_switch(
      data = fp16::builtin_mul(a._data, b._data),
      data = __acpp_sscp_half_mul(a._data, b._data),
      data = fp16::create(__hmul(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin mul for native _Float16 type
      data = fp16::builtin_mul(a._data, b._data));
    return detail::create_half(data);
  }

  ACPP_UNIVERSAL_TARGET
  friend half operator/(const half& a, const half& b) noexcept {
    fp16::half_storage data;
    __acpp_backend_switch(
      data = fp16::builtin_div(a._data, b._data),
      data = __acpp_sscp_half_div(a._data, b._data),
      data = fp16::create(__hdiv(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data))),
      // HIP uses compiler builtin div for native _Float16 type
      data = fp16::builtin_div(a._data, b._data));
    return detail::create_half(data);
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

    // operator +,-,*,/ for combinations of half and other types
#define OP_FOR_TYPE(op, type)                                         \
  friend half operator op(const half lhs, const type rhs) {           \
    return lhs op half(rhs);                                          \
  }                                                                   \
                                                                      \
  friend half operator op(const type lhs, const half rhs) {           \
    return half(lhs) op rhs;                                          \
  }

#define OP(op)                                                        \
  OP_FOR_TYPE(op, int)                                                \
  OP_FOR_TYPE(op, unsigned int)                                       \
  OP_FOR_TYPE(op, long)                                               \
  OP_FOR_TYPE(op, long long)                                          \
  OP_FOR_TYPE(op, unsigned long)                                      \
  OP_FOR_TYPE(op, unsigned long long)                                 \
  OP_FOR_TYPE(op, float)                                              \
  OP_FOR_TYPE(op, double)

  OP(+)
  OP(-)
  OP(*)
  OP(/)

#undef OP
#undef OP_FOR_TYPE

  friend bool operator==(const half& a, const half& b) noexcept {
    return a._data == b._data;
  }

  friend bool operator!=(const half& a, const half& b) noexcept {
    return a._data != b._data;
  }

  // Operator is +/- unary
  friend half& operator+(half& a) noexcept {
    return a;
  }

  friend half& operator-(half& a) noexcept {
    constexpr __acpp_uint16 sign_mask = 0x8000;
    a._data ^= sign_mask;
    return a; 
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator<(const half& a, const half& b) noexcept {
    __acpp_backend_switch(
      return fp16::builtin_less_than(a._data, b._data),
      return __acpp_sscp_half_lt(a._data, b._data),
      return __hlt(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_less_than(a._data, b._data))
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator<=(const half& a, const half& b) noexcept {
    __acpp_backend_switch(
      return fp16::builtin_less_than_equal(a._data, b._data),
      return __acpp_sscp_half_lte(a._data, b._data),
      return __hle(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_less_than_equal(a._data, b._data))
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator>(const half& a, const half& b) noexcept {
    __acpp_backend_switch(
      return fp16::builtin_greater_than(a._data, b._data),
      return __acpp_sscp_half_gt(a._data, b._data),
      return __hgt(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_greater_than(a._data, b._data))
  }

  ACPP_UNIVERSAL_TARGET
  friend bool operator>=(const half& a, const half& b) noexcept {
    __acpp_backend_switch(
      return fp16::builtin_greater_than_equal(a._data, b._data),
      return __acpp_sscp_half_gte(a._data, b._data),
      return __hge(fp16::as_cuda_half(a._data), fp16::as_cuda_half(b._data)),
      return fp16::builtin_greater_than_equal(a._data, b._data))
  }
private:
  fp16::half_storage _data;
};

namespace detail {
  constexpr half create_half(fp16::half_storage h) {
    half v;
    v._data = h;
    return v;
  }
  constexpr fp16::half_storage get_half_storage(half h) {
    return h._data;
  }
}

}
}

namespace std {
  template<> class numeric_limits<hipsycl::sycl::half>{
  public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_bounded = true;
    static constexpr bool is_iec559 = true;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr bool has_denorm_loss = false;
    static constexpr bool traps = false;
    static constexpr bool tinyness_before = false;
    static constexpr float_round_style round_style = float_round_style::round_indeterminate;
    static constexpr int digits = 11;
    static constexpr int digits10 = 3;
    static constexpr int max_digits10 = 5;
    static constexpr int radix = 2;
    static constexpr int min_exponent = -13;
    static constexpr int min_exponent10 = -4;
    static constexpr int max_exponent = 16;
    static constexpr int max_exponent10 = 4;

    static constexpr hipsycl::sycl::half min() noexcept {
      return hipsycl::sycl::detail::create_half(0x0400);
    }
    static constexpr hipsycl::sycl::half lowest() noexcept {
      return hipsycl::sycl::detail::create_half(0xFBFF);
    }
    static constexpr hipsycl::sycl::half max() noexcept {
      return hipsycl::sycl::detail::create_half(0x7BFF);
    }
    static constexpr hipsycl::sycl::half epsilon() noexcept {
      return hipsycl::sycl::detail::create_half(0x1400);
    }
    static constexpr hipsycl::sycl::half round_error() noexcept {
      return hipsycl::sycl::detail::create_half(
          (round_style == std::round_to_nearest) ? 0x3800 : 0x3C00);
    }
    static constexpr hipsycl::sycl::half infinity() noexcept {
      return hipsycl::sycl::detail::create_half(0x7C00);
    }
    static constexpr hipsycl::sycl::half quiet_NaN() noexcept {
      return hipsycl::sycl::detail::create_half(0x7FFF);
    }
    static constexpr hipsycl::sycl::half signaling_NaN() noexcept {
      return hipsycl::sycl::detail::create_half(0x7DFF);
    }
    static constexpr hipsycl::sycl::half denorm_min() noexcept {
      return hipsycl::sycl::detail::create_half(0x0001);
    }
  };

  template <> struct hash<hipsycl::sycl::half> {
    size_t operator()(hipsycl::sycl::half h) const {
      auto data = hipsycl::sycl::detail::get_half_storage(h);
      return hash<hipsycl::fp16::half_storage>{}(data);
    }
  };
}

#endif
