
/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_BUILTIN_CONFIG_HPP
#define HIPSYCL_SSCP_BUILTIN_CONFIG_HPP

#include "../../memory.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"
#include "hipSYCL/sycl/libkernel/detail/int_types.hpp"

#define HIPSYCL_SSCP_BUILTIN_ATTRIBUTES __attribute__((always_inline))
#define HIPSYCL_SSCP_BUILTIN_DEFAULT_LINKAGE extern "C"
#define HIPSYCL_SSCP_BUILTIN HIPSYCL_SSCP_BUILTIN_DEFAULT_LINKAGE HIPSYCL_SSCP_BUILTIN_ATTRIBUTES
#define HIPSYCL_SSCP_CONVERGENT_BUILTIN HIPSYCL_SSCP_BUILTIN __attribute__((convergent))

using __hipsycl_sscp_memory_scope = hipsycl::sycl::memory_scope;
using __hipsycl_sscp_memory_order = hipsycl::sycl::memory_order;
using __hipsycl_sscp_address_space = hipsycl::sycl::access::address_space;

enum class __hipsycl_sscp_algorithm_op : __hipsycl_int32 {
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


// Note: __hipsycl_native_f16 should only be used for backend-specific SSCP builtin
// interfaces, e.g. to declare CUDA libdevice functions or AMD ocml builtins. It
// should not be used in generic hipSYCL headers - those should use
// hipsycl::fp16::half_storage/__hipsycl_f16 of sycl::half instead.
using __hipsycl_native_f16 = hipsycl::fp16::native_t;
using __hipsycl_f16 = hipsycl::fp16::half_storage;
using __hipsycl_f32 = float;
using __hipsycl_f64 = double;

#define HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(base_type) \
    using base_type##_2 = base_type __attribute__((ext_vector_type(2))); \
    using base_type##_4 = base_type __attribute__((ext_vector_type(4))); \

HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_native_f16)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_f32)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_f64)
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_int16);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_uint16);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_int32);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_uint32);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_int64);
HIPSYCL_SSCP_BUILTIN_CONFIG_DECLARE_VEC_TYPES(__hipsycl_uint64);



#endif
