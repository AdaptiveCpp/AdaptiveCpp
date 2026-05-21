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

#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/broadcast.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/broadcast.hpp"

#include "helpers.hpp"

#include <type_traits>

using namespace hipsycl::sycl::detail::metal_builtins;

#define BROADCAST_TYPES \
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

#define X(type) HIPSYCL_SSCP_BUILTIN type __acpp_sscp_metal_broadcast_##type(const char* s, type value, i32 local_id);
BROADCAST_TYPES
#undef X

template <typename T>
inline T __acpp_sscp_metal_broadcast(T value, i32 local_id) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_broadcast_##type("simd_broadcast", value, local_id); \
  }

  BROADCAST_TYPES

#undef X
}

template<typename T>
inline T __acpp_sscp_sub_group_broadcast(i32 local_id, T value) {
  if constexpr(sizeof(T) <= 4) {
    return __acpp_sscp_metal_broadcast(value, local_id);
  } else {
    // For 64-bit types, split into two 32-bit parts
    union {
      T value;
      u32 parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (uint i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = __acpp_sscp_metal_broadcast(in.parts[i], local_id);
    }
    return out.value;
  }
}

#define ACPP_SUBGROUP_BROADCAST(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_broadcast_##type(i32 sender, type x) { \
    return __acpp_sscp_sub_group_broadcast(sender, x); \
}

ACPP_SUBGROUP_BROADCAST(i8)
ACPP_SUBGROUP_BROADCAST(i16)
ACPP_SUBGROUP_BROADCAST(i32)
ACPP_SUBGROUP_BROADCAST(i64)
ACPP_SUBGROUP_BROADCAST(u8)
ACPP_SUBGROUP_BROADCAST(u16)
ACPP_SUBGROUP_BROADCAST(u32)
ACPP_SUBGROUP_BROADCAST(u64)
ACPP_SUBGROUP_BROADCAST(f16)
ACPP_SUBGROUP_BROADCAST(f32)

template<typename T>
inline T __acpp_sscp_work_group_broadcast(i32 local_id, T value) {
  __attribute__((address_space(3))) T* scratch = __acpp_sscp_get_typed_dynamic_local_memory<T>();
  return hipsycl::libkernel::sscp::wg_broadcast(local_id, value, scratch);
}

#define ACPP_WORKGROUP_BROADCAST(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_work_group_broadcast_##type(i32 sender, type x) { \
    return __acpp_sscp_work_group_broadcast(sender, x); \
}

ACPP_WORKGROUP_BROADCAST(i8)
ACPP_WORKGROUP_BROADCAST(i16)
ACPP_WORKGROUP_BROADCAST(i32)
ACPP_WORKGROUP_BROADCAST(i64)
ACPP_WORKGROUP_BROADCAST(u8)
ACPP_WORKGROUP_BROADCAST(u16)
ACPP_WORKGROUP_BROADCAST(u32)
ACPP_WORKGROUP_BROADCAST(u64)
ACPP_WORKGROUP_BROADCAST(f16)
ACPP_WORKGROUP_BROADCAST(f32)
