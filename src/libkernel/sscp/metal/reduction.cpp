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
#include "hipSYCL/sycl/libkernel/sscp/builtins/reduction.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/localmem.hpp"

#include "helpers.hpp"

#include <limits>
#include <type_traits>

using namespace hipsycl::sycl::detail::metal_builtins;

#define REDUCTION_INT_TYPES \
  X(bool) \
  X(i8) \
  X(i16) \
  X(i32) \
  X(i64) \
  X(u8) \
  X(u16) \
  X(u32) \
  X(u64)

#define REDUCTION_FLOAT_TYPES \
  X(f16) \
  X(f32)

#define REDUCTION_TYPES \
  REDUCTION_INT_TYPES \
  REDUCTION_FLOAT_TYPES
#define X(type) HIPSYCL_SSCP_BUILTIN type __acpp_sscp_metal_reduce_##type(const char* s, type value);
REDUCTION_TYPES
#undef X

template <typename T>
inline T __acpp_sscp_metal_reduce_(const char* s, T value) {
#define X(type) \
  if constexpr(std::is_same_v<T, type>) { \
    return __acpp_sscp_metal_reduce_##type(s, value); \
  }

  REDUCTION_TYPES

#undef X
}

template <__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_sub_group_reduce(T value) {
  if constexpr (op == __acpp_sscp_algorithm_op::plus) {
    return __acpp_sscp_metal_reduce_<T>("simd_sum", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::multiply) {
    return __acpp_sscp_metal_reduce_<T>("simd_product", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::min && std::is_integral_v<T> && std::is_signed_v<T>) {
    return __acpp_sscp_metal_reduce_<T>("simd_min(__as_signed(%s))", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::min) {
    return __acpp_sscp_metal_reduce_<T>("simd_min", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::max && std::is_integral_v<T> && std::is_signed_v<T>) {
    return __acpp_sscp_metal_reduce_<T>("simd_max(__as_signed(%s))", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::max) {
    return __acpp_sscp_metal_reduce_<T>("simd_max", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::bit_and) {
    return __acpp_sscp_metal_reduce_<T>("simd_and", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::bit_or) {
    return __acpp_sscp_metal_reduce_<T>("simd_or", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::bit_xor) {
    return __acpp_sscp_metal_reduce_<T>("simd_xor", value);
  } else if constexpr (op == __acpp_sscp_algorithm_op::logical_and) {
    return static_cast<T>(__acpp_sscp_metal_reduce_<bool>("simd_all", static_cast<bool>(value)));
  } else if constexpr (op == __acpp_sscp_algorithm_op::logical_or) {
    return static_cast<T>(__acpp_sscp_metal_reduce_<bool>("simd_any", static_cast<bool>(value)));
  }
}

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op, type value) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::plus>(value); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::multiply>(value); \
  case __acpp_sscp_algorithm_op::min: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::min>(value); \
  case __acpp_sscp_algorithm_op::max: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::max>(value); \
  case __acpp_sscp_algorithm_op::bit_and: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::bit_and>(value); \
  case __acpp_sscp_algorithm_op::bit_or: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::bit_or>(value); \
  case __acpp_sscp_algorithm_op::bit_xor: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::bit_xor>(value); \
  case __acpp_sscp_algorithm_op::logical_and: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::logical_and>(value); \
  case __acpp_sscp_algorithm_op::logical_or: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::logical_or>(value); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}
REDUCTION_INT_TYPES
#undef X

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_sub_group_reduce_##type(__acpp_sscp_algorithm_op op, type value) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::plus>(value); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::multiply>(value); \
  case __acpp_sscp_algorithm_op::min: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::min>(value); \
  case __acpp_sscp_algorithm_op::max: \
    return __acpp_sscp_sub_group_reduce<__acpp_sscp_algorithm_op::max>(value); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}
REDUCTION_FLOAT_TYPES
#undef X

template<__acpp_sscp_algorithm_op op, typename T>
inline T __acpp_sscp_work_group_reduce(T value) {
  auto* scratch = __acpp_sscp_get_typed_dynamic_local_memory<T>();
  const uint subgroup_size = __acpp_sscp_get_subgroup_max_size();
  const uint lid = __acpp_sscp_typed_get_local_linear_id<3, uint>();
  const uint local_size = __acpp_sscp_typed_get_local_size<3, uint>();

  const uint group_id = __acpp_sscp_get_subgroup_id();
  const uint lane_id  = __acpp_sscp_get_subgroup_local_id();
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  auto reduce_op = [&](T v) {
    return __acpp_sscp_sub_group_reduce<op>(v);
  };

  using binary_op_type = typename hipsycl::libkernel::sscp::get_op<op>::type;
  auto binary_op = [](T a, T b) -> T { return binary_op_type{}(a, b); };
  auto identity = [&]() -> T { return __acpp_sscp_metal_op_init_value<op, T>(); };

  const uint group_base = group_id * subgroup_size;
  uint active = 0;
  if(group_base < local_size) {
    uint rem = local_size - group_base;
    active = (rem < subgroup_size) ? rem : subgroup_size;
  }

  T v = value;
  if(lane_id >= active) v = identity();

  const T sg_reduced = reduce_op(v);

  if(lane_id == 0) scratch[group_id] = sg_reduced;
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

  if(group_id == 0) {
    T result = identity();

    if(ngroups <= subgroup_size) {
      T x = identity();
      if(lid < ngroups) x = scratch[lid];
      result = reduce_op(x);

      if(lane_id == 0) scratch[0] = result;
    }
    else if(ngroups <= 2u * subgroup_size) {
      T x0 = identity();
      if(lid < subgroup_size) {
        uint i = lid;
        if(i < ngroups) x0 = scratch[i];
      }
      T r0 = reduce_op(x0);

      T x1 = identity();
      if(lid < subgroup_size) {
        uint i = lid + subgroup_size;
        if(i < ngroups) x1 = scratch[i];
      }
      T r1 = reduce_op(x1);

      if(lane_id == 0) scratch[0] = r0;
      if(lane_id == 1) scratch[1] = r1;
      __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);

      T y = identity();
      if(lid < 2) y = scratch[lid];
      result = reduce_op(y);

      if(lane_id == 0) scratch[0] = result;
    }
    else {
      uint active_ngroups = ngroups;
      while(active_ngroups > 1) {
        uint offset = (active_ngroups + 1u) >> 1;
        if(lid < offset) {
          uint j = lid + offset;
          if(j < active_ngroups)
            scratch[lid] = binary_op(scratch[lid], scratch[j]);
        }
        __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
        active_ngroups = offset;
      }
      if(lane_id == 0) scratch[0] = scratch[0];
    }
  }

  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  T tmp = scratch[0];
  __acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed);
  return tmp;
}

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_work_group_reduce_##type(__acpp_sscp_algorithm_op op, type value) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::plus>(value); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::multiply>(value); \
  case __acpp_sscp_algorithm_op::min: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::min>(value); \
  case __acpp_sscp_algorithm_op::max: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::max>(value); \
  case __acpp_sscp_algorithm_op::bit_and: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::bit_and>(value); \
  case __acpp_sscp_algorithm_op::bit_or: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::bit_or>(value); \
  case __acpp_sscp_algorithm_op::bit_xor: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::bit_xor>(value); \
  case __acpp_sscp_algorithm_op::logical_and: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::logical_and>(value); \
  case __acpp_sscp_algorithm_op::logical_or: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::logical_or>(value); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}
REDUCTION_INT_TYPES
#undef X

#define X(type) \
HIPSYCL_SSCP_CONVERGENT_BUILTIN type __acpp_sscp_work_group_reduce_##type(__acpp_sscp_algorithm_op op, type value) { \
  switch (op) { \
  case __acpp_sscp_algorithm_op::plus: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::plus>(value); \
  case __acpp_sscp_algorithm_op::multiply: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::multiply>(value); \
  case __acpp_sscp_algorithm_op::min: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::min>(value); \
  case __acpp_sscp_algorithm_op::max: \
    return __acpp_sscp_work_group_reduce<__acpp_sscp_algorithm_op::max>(value); \
  default: \
    __builtin_trap(); \
    return 0;\
  } \
}
REDUCTION_FLOAT_TYPES
#undef X
