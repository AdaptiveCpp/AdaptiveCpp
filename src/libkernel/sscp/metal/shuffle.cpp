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

#include "hipSYCL/sycl/libkernel/sscp/builtins/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

#include "shuffle_helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

template<typename T>
inline T __acpp_sscp_sub_group_shl(T value, u32 shift) {
  if constexpr(sizeof(T) <= 4) {
    return __acpp_sscp_metal_shuffle_down(value, shift);
  } else {
    // For 64-bit types, split into two 32-bit parts
    union {
      T value;
      u32 parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (uint i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = __acpp_sscp_metal_shuffle_down(in.parts[i], shift);
    }
    return out.value;
  }
}

#define ACPP_SUBGROUP_SHL(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_shl_##type(type value, u32 delta) { \
    return __acpp_sscp_sub_group_shl(value, delta); \
}

ACPP_SUBGROUP_SHL(i8)
ACPP_SUBGROUP_SHL(i16)
ACPP_SUBGROUP_SHL(i32)
ACPP_SUBGROUP_SHL(i64)

template<typename T>
inline T __acpp_sscp_sub_group_shr(T value, u32 shift) {
  if constexpr(sizeof(T) <= 4) {
    return __acpp_sscp_metal_shuffle_up(value, shift);
  } else {
    // For 64-bit types, split into two 32-bit parts
    union {
      T value;
      u32 parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (uint i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = __acpp_sscp_metal_shuffle_up(in.parts[i], shift);
    }
    return out.value;
  }
}

#define ACPP_SUBGROUP_SHR(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_shr_##type(type value, u32 delta) { \
    return __acpp_sscp_sub_group_shr(value, delta); \
}

ACPP_SUBGROUP_SHR(i8)
ACPP_SUBGROUP_SHR(i16)
ACPP_SUBGROUP_SHR(i32)
ACPP_SUBGROUP_SHR(i64)

template<typename T>
inline T __acpp_sscp_sub_group_permute(T value, i32 mask) {
  if constexpr(sizeof(T) <= 4) {
    return __acpp_sscp_metal_shuffle_xor(value, mask);
  } else {
    // For 64-bit types, split into two 32-bit parts
    union {
      T value;
      u32 parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (uint i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = __acpp_sscp_metal_shuffle_xor(in.parts[i], mask);
    }
    return out.value;
  }
}

#define ACPP_SUBGROUP_PERMUTE(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_permute_##type(type value, i32 mask) { \
    return __acpp_sscp_sub_group_permute(value, mask); \
}

ACPP_SUBGROUP_PERMUTE(i8)
ACPP_SUBGROUP_PERMUTE(i16)
ACPP_SUBGROUP_PERMUTE(i32)
ACPP_SUBGROUP_PERMUTE(i64)

template<typename T>
inline T __acpp_sscp_sub_group_select(T value, i32 lane) {
  if constexpr(sizeof(T) <= 4) {
    return __acpp_sscp_metal_shuffle(value, lane);
  } else {
    // For 64-bit types, split into two 32-bit parts
    union {
      T value;
      u32 parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (uint i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = __acpp_sscp_metal_shuffle(in.parts[i], lane);
    }
    return out.value;
  }
}

#define ACPP_SUBGROUP_SELECT(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_select_##type(type value, i32 id) { \
    return __acpp_sscp_sub_group_select(value, id); \
}

ACPP_SUBGROUP_SELECT(i8)
ACPP_SUBGROUP_SELECT(i16)
ACPP_SUBGROUP_SELECT(i32)
ACPP_SUBGROUP_SELECT(i64)
