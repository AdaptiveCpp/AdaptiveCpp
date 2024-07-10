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
#ifndef HIPSYCL_SSCP_BUILTIN_CONFIG_HPP
#define HIPSYCL_SSCP_BUILTIN_CONFIG_HPP

#include "../../memory.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"

#define HIPSYCL_SSCP_BUILTIN_ATTRIBUTES __attribute__((always_inline))
#define HIPSYCL_SSCP_BUILTIN_DEFAULT_LINKAGE extern "C"
#define HIPSYCL_SSCP_BUILTIN HIPSYCL_SSCP_BUILTIN_DEFAULT_LINKAGE HIPSYCL_SSCP_BUILTIN_ATTRIBUTES
#define HIPSYCL_SSCP_CONVERGENT_BUILTIN HIPSYCL_SSCP_BUILTIN __attribute__((convergent))

using __acpp_sscp_memory_scope = hipsycl::sycl::memory_scope;
using __acpp_sscp_memory_order = hipsycl::sycl::memory_order;
using __acpp_sscp_address_space = hipsycl::sycl::access::address_space;

enum class __acpp_sscp_algorithm_op : __acpp_int32 {
  plus,
  multiply,
  min,
  max,
  bit_and,
  bit_or,
  bit_xor,
  logical_and,
  logical_or
};


// Note: __acpp_native_f16 should only be used for backend-specific SSCP builtin
// interfaces, e.g. to declare CUDA libdevice functions or AMD ocml builtins. It
// should not be used in generic hipSYCL headers - those should use
// hipsycl::fp16::half_storage/__acpp_f16 of sycl::half instead.
using __acpp_native_f16 = hipsycl::fp16::native_t;
using __acpp_f16 = hipsycl::fp16::half_storage;
using __acpp_f32 = float;
using __acpp_f64 = double;

#define HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(base_type) \
    using base_type##_2 = base_type __attribute__((ext_vector_type(2))); \
    using base_type##_4 = base_type __attribute__((ext_vector_type(4))); \

HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_native_f16)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_f32)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_f64)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_int16);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_uint16);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_int32);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_uint32);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_int64);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__acpp_uint64);



#endif
