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

#ifndef HIPSYCL_METAL_HELPERS_HPP
#define HIPSYCL_METAL_HELPERS_HPP

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"

namespace hipsycl {
namespace sycl {
namespace detail::metal_builtins {

using i8 = __acpp_int8;
using i16 = __acpp_int16;
using i32 = __acpp_int32;
using i64 = __acpp_int64;

using u8 = __acpp_uint8;
using u16 = __acpp_uint16;
using u32 = __acpp_uint32;
using u64 = __acpp_uint64;

using f16 = __acpp_f16;
using f32 = float;
using i1 = bool;

using uint = __acpp_uint32;
using uchar = __acpp_uint8;

HIPSYCL_SSCP_BUILTIN u32 __acpp_sscp_get_dynamic_local_memory_size(); // size of local memory "user" region, i.e. the part that is not used for workgroup synchronization or other internal purposes

template<typename T> inline __attribute__((address_space(3))) T* __acpp_sscp_get_typed_dynamic_local_memory() {
  __attribute__((address_space(3))) uchar* ptr = (__attribute__((address_space(3))) uchar*)__acpp_sscp_get_dynamic_local_memory();
  ptr += __acpp_sscp_get_dynamic_local_memory_size();
  __attribute__((address_space(3))) T* typed_ptr = (__attribute__((address_space(3))) T*)ptr;
  return typed_ptr;
}


}
}
}

#endif