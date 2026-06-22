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
#include "hipSYCL/sycl/libkernel/sscp/builtins/detail/shuffle.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"

// clspv doesn't support OpenCL-C cl_khr_subgroup_shuffle_relative builtins
// `sub_group_shuffle_down` and `sub_group_shuffle_up` so we can't use these
// in the implementation

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shl_i8(__acpp_int8 value,
                                         __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id + delta;
  if (target_id >= __acpp_sscp_get_subgroup_size())
    target_id = local_id;
  return __acpp_sscp_sub_group_select_i8(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shl_i16(__acpp_int16 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id + delta;
  if (target_id >= __acpp_sscp_get_subgroup_size())
    target_id = local_id;
  return __acpp_sscp_sub_group_select_i16(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shl_i32(__acpp_int32 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id + delta;
  if (target_id >= __acpp_sscp_get_subgroup_size())
    target_id = local_id;
  return __acpp_sscp_sub_group_select_i32(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shl_i64(__acpp_int64 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id + delta;
  if (target_id >= __acpp_sscp_get_subgroup_size())
    target_id = local_id;
  return __acpp_sscp_sub_group_select_i64(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_shr_i8(__acpp_int8 value,
                                         __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id;
  if (local_id >= delta)
    target_id -= delta;
  return __acpp_sscp_sub_group_select_i8(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_shr_i16(__acpp_int16 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id;
  if (local_id >= delta)
    target_id -= delta;
  return __acpp_sscp_sub_group_select_i16(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_shr_i32(__acpp_int32 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id;
  if (local_id >= delta)
    target_id -= delta;
  return __acpp_sscp_sub_group_select_i32(value, target_id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_shr_i64(__acpp_int64 value,
                                           __acpp_uint32 delta) {
  __acpp_int32 local_id = __acpp_sscp_get_subgroup_local_id();
  __acpp_int32 target_id = local_id;
  if (local_id >= delta)
    target_id -= delta;
  return __acpp_sscp_sub_group_select_i64(value, target_id);
}

// OpenCL-C cl_khr_subgroup_shuffle declarations
__acpp_int8 sub_group_shuffle(__acpp_int8 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int16 sub_group_shuffle(__acpp_int16 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int32 sub_group_shuffle(__acpp_int32 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int64 sub_group_shuffle(__acpp_int64 value, __acpp_uint32 delta)
    __attribute__((convergent));

__acpp_int8 sub_group_shuffle_xor(__acpp_int8 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int16 sub_group_shuffle_xor(__acpp_int16 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int32 sub_group_shuffle_xor(__acpp_int32 value, __acpp_uint32 delta)
    __attribute__((convergent));
__acpp_int64 sub_group_shuffle_xor(__acpp_int64 value, __acpp_uint32 delta)
    __attribute__((convergent));

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_permute_i8(__acpp_int8 value,
                                             __acpp_int32 mask) {
  return sub_group_shuffle_xor(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_permute_i16(__acpp_int16 value,
                                               __acpp_int32 mask) {
  return sub_group_shuffle_xor(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_permute_i32(__acpp_int32 value,
                                               __acpp_int32 mask) {
  return sub_group_shuffle_xor(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_permute_i64(__acpp_int64 value,
                                               __acpp_int32 mask) {
  return sub_group_shuffle_xor(value, mask);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int8 __acpp_sscp_sub_group_select_i8(__acpp_int8 value,
                                            __acpp_int32 id) {
  return sub_group_shuffle(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int16 __acpp_sscp_sub_group_select_i16(__acpp_int16 value,
                                              __acpp_int32 id) {
  return sub_group_shuffle(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int32 __acpp_sscp_sub_group_select_i32(__acpp_int32 value,
                                              __acpp_int32 id) {
  return sub_group_shuffle(value, id);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__acpp_int64 __acpp_sscp_sub_group_select_i64(__acpp_int64 value,
                                              __acpp_int32 id) {
  return sub_group_shuffle(value, id);
}
