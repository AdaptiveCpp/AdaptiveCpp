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


#ifndef ACPP_PCUDA_ATOMIC_HPP
#define ACPP_PCUDA_ATOMIC_HPP

#include "builtin_call.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"

#define __PCUDA_DECLARE_ATOMIC_ADD(type, builtin)                              \
  inline type atomicAdd(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::device, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicAdd_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::system, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicAdd_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::work_group, address, val),           \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_ADD(int, __acpp_sscp_atomic_fetch_add_i32)
__PCUDA_DECLARE_ATOMIC_ADD(unsigned int, __acpp_sscp_atomic_fetch_add_u32)
__PCUDA_DECLARE_ATOMIC_ADD(unsigned long long int, __acpp_sscp_atomic_fetch_add_u64)
__PCUDA_DECLARE_ATOMIC_ADD(float, __acpp_sscp_atomic_fetch_add_f32)
__PCUDA_DECLARE_ATOMIC_ADD(double, __acpp_sscp_atomic_fetch_add_f64)

#define __PCUDA_DECLARE_ATOMIC_SUB(type, builtin)                              \
  inline type atomicSub(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::device, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicSub_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::system, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicSub_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::work_group, address, val),           \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_SUB(int, __acpp_sscp_atomic_fetch_sub_i32)
__PCUDA_DECLARE_ATOMIC_SUB(unsigned int, __acpp_sscp_atomic_fetch_sub_u32)

#define __PCUDA_DECLARE_ATOMIC_EXCH(type, builtin_type, builtin)               \
  inline type atomicExch(type *address, type val) {                            \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::device,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicExch_system(type *address, type val) {                     \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::system,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicExch_block(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::work_group,       \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_EXCH(int, __acpp_int32, __acpp_sscp_atomic_exchange_i32)
__PCUDA_DECLARE_ATOMIC_EXCH(unsigned int, __acpp_int32,
                            __acpp_sscp_atomic_exchange_i32)
__PCUDA_DECLARE_ATOMIC_EXCH(unsigned long long int, __acpp_int64,
                            __acpp_sscp_atomic_exchange_i64)
__PCUDA_DECLARE_ATOMIC_EXCH(float, __acpp_int32, __acpp_sscp_atomic_exchange_i32)

#define __PCUDA_DECLARE_ATOMIC_MIN(type, builtin)                              \
  inline type atomicMin(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::device, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicMin_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::system, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicMin_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::work_group, address, val),           \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_MIN(int, __acpp_sscp_atomic_fetch_min_i32)
__PCUDA_DECLARE_ATOMIC_MIN(unsigned int, __acpp_sscp_atomic_fetch_min_u32)
__PCUDA_DECLARE_ATOMIC_MIN(unsigned long long int,
                           __acpp_sscp_atomic_fetch_min_u64)
__PCUDA_DECLARE_ATOMIC_MIN(long long int, __acpp_sscp_atomic_fetch_min_i64)

#define __PCUDA_DECLARE_ATOMIC_MAX(type, builtin)                              \
  inline type atomicMax(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::device, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicMax_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::system, address, val),               \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicMax_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        builtin(__acpp_sscp_address_space::generic_space,                      \
                __acpp_sscp_memory_order::relaxed,                             \
                __acpp_sscp_memory_scope::work_group, address, val),           \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_MAX(int, __acpp_sscp_atomic_fetch_max_i32)
__PCUDA_DECLARE_ATOMIC_MAX(unsigned int, __acpp_sscp_atomic_fetch_max_u32)
__PCUDA_DECLARE_ATOMIC_MAX(unsigned long long int,
                           __acpp_sscp_atomic_fetch_max_u64)
__PCUDA_DECLARE_ATOMIC_MAX(long long int, __acpp_sscp_atomic_fetch_max_i64)

#define __PCUDA_DECLARE_ATOMIC_AND(type, builtin_type, builtin)                \
  inline type atomicAnd(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::device,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicAnd_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::system,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicAnd_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::work_group,       \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_AND(int, __acpp_int32, __acpp_sscp_atomic_fetch_and_i32)
__PCUDA_DECLARE_ATOMIC_AND(unsigned int, __acpp_int32,
                           __acpp_sscp_atomic_fetch_and_i32)
__PCUDA_DECLARE_ATOMIC_AND(unsigned long long int, __acpp_int64,
                           __acpp_sscp_atomic_fetch_and_i64)

#define __PCUDA_DECLARE_ATOMIC_OR(type, builtin_type, builtin)                 \
  inline type atomicOr(type *address, type val) {                              \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::device,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicOr_system(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::system,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicOr_block(type *address, type val) {                        \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::work_group,       \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_OR(int, __acpp_int32, __acpp_sscp_atomic_fetch_or_i32)
__PCUDA_DECLARE_ATOMIC_OR(unsigned int, __acpp_int32,
                          __acpp_sscp_atomic_fetch_or_i32)
__PCUDA_DECLARE_ATOMIC_OR(unsigned long long int, __acpp_int64,
                          __acpp_sscp_atomic_fetch_or_i64)

#define __PCUDA_DECLARE_ATOMIC_XOR(type, builtin_type, builtin)                \
  inline type atomicXor(type *address, type val) {                             \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::device,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicXor_system(type *address, type val) {                      \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::system,           \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }                                                                            \
  inline type atomicXor_block(type *address, type val) {                       \
    return PCUDA_BUILTIN_CALL_RESULT(                                          \
        __builtin_bit_cast(type,                                               \
                           builtin(__acpp_sscp_address_space::generic_space,   \
                                   __acpp_sscp_memory_order::relaxed,          \
                                   __acpp_sscp_memory_scope::work_group,       \
                                   (builtin_type *)address,                    \
                                   __builtin_bit_cast(builtin_type, val))),    \
        static_cast<type>(0));                                                 \
  }

__PCUDA_DECLARE_ATOMIC_XOR(int, __acpp_int32, __acpp_sscp_atomic_fetch_xor_i32)
__PCUDA_DECLARE_ATOMIC_XOR(unsigned int, __acpp_int32,
                           __acpp_sscp_atomic_fetch_xor_i32)
__PCUDA_DECLARE_ATOMIC_XOR(unsigned long long int, __acpp_int64,
                           __acpp_sscp_atomic_fetch_xor_i64)

#define __PCUDA_DECLARE_ATOMIC_CAS(type, builtin_type, builtin)                \
  inline type atomicCAS(type *address, type compare, type val) {               \
    if (__acpp_sscp_is_device) {                                               \
      builtin_type expected = __builtin_bit_cast(builtin_type, compare);       \
      builtin(__acpp_sscp_address_space::generic_space,                        \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_scope::device, (builtin_type *)address,       \
              &expected, __builtin_bit_cast(builtin_type, val));               \
      return __builtin_bit_cast(type, expected);                               \
    }                                                                          \
    return val;                                                                \
  }                                                                            \
  inline type atomicCAS_system(type *address, type compare, type val) {        \
    if (__acpp_sscp_is_device) {                                               \
      builtin_type expected = __builtin_bit_cast(builtin_type, compare);       \
      builtin(__acpp_sscp_address_space::generic_space,                        \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_scope::system, (builtin_type *)address,       \
              &expected, __builtin_bit_cast(builtin_type, val));               \
      return __builtin_bit_cast(type, expected);                               \
    }                                                                          \
    return val;                                                                \
  }                                                                            \
  inline type atomicCAS_block(type *address, type compare, type val) {         \
    if (__acpp_sscp_is_device) {                                               \
      builtin_type expected = __builtin_bit_cast(builtin_type, compare);       \
      builtin(__acpp_sscp_address_space::generic_space,                        \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_order::relaxed,                               \
              __acpp_sscp_memory_scope::work_group, (builtin_type *)address,   \
              &expected, __builtin_bit_cast(builtin_type, val));               \
      return __builtin_bit_cast(type, expected);                               \
    }                                                                          \
    return val;                                                                \
  }

__PCUDA_DECLARE_ATOMIC_CAS(int, __acpp_int32, __acpp_sscp_cmp_exch_strong_i32)
__PCUDA_DECLARE_ATOMIC_CAS(unsigned int, __acpp_int32,
                           __acpp_sscp_cmp_exch_strong_i32)
__PCUDA_DECLARE_ATOMIC_CAS(unsigned long long int, __acpp_int64,
                           __acpp_sscp_cmp_exch_strong_i64)
__PCUDA_DECLARE_ATOMIC_CAS(unsigned short int, __acpp_int16,
                           __acpp_sscp_cmp_exch_strong_i16)
namespace __pcuda_detail {
inline unsigned int atomicInc(unsigned int *address, unsigned int val,
                              __acpp_sscp_memory_scope scope) {
  //Computes ((old >= val) ? 0 : (old+1))
  if(__acpp_sscp_is_device) {
    __acpp_sscp_memory_order order = __acpp_sscp_memory_order::relaxed;
    __acpp_sscp_address_space as = __acpp_sscp_address_space::generic_space;

    __acpp_int32 *i_address = (__acpp_int32 *)address;

    __acpp_int32 old = __acpp_sscp_atomic_load_i32(as, order, scope, i_address);
    __acpp_int32 desired = old;
    do {
      unsigned int current_old = __builtin_bit_cast(unsigned int, old);
      desired = (current_old >= val) ? 0 : __builtin_bit_cast(__acpp_int32, current_old + 1);
    } while (!__acpp_sscp_cmp_exch_strong_i32(as, order, order, scope, i_address, &old, desired));
    return __builtin_bit_cast(unsigned int, old);
  }
  return val;
}


inline unsigned int atomicDec(unsigned int *address, unsigned int val,
                              __acpp_sscp_memory_scope scope) {
  //Computes (((old == 0) || (old > val)) ? val : (old-1)
  if(__acpp_sscp_is_device) {
    __acpp_sscp_memory_order order = __acpp_sscp_memory_order::relaxed;
    __acpp_sscp_address_space as = __acpp_sscp_address_space::generic_space;

    __acpp_int32 *i_address = (__acpp_int32 *)address;
    __acpp_int32  i_val = __builtin_bit_cast(__acpp_int32, val);

    __acpp_int32 old = __acpp_sscp_atomic_load_i32(as, order, scope, i_address);
    __acpp_int32 desired = old;
    do {
      unsigned int current_old = __builtin_bit_cast(unsigned int, old);
      desired = ((current_old == 0) || (current_old > val))
                    ? i_val
                    : __builtin_bit_cast(__acpp_int32, current_old - 1);
    } while (!__acpp_sscp_cmp_exch_strong_i32(as, order, order, scope, i_address, &old, desired));
    return __builtin_bit_cast(unsigned int, old);
  }
  return val;
}

} // namespace __pcuda_detail

inline unsigned int atomicInc(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicInc(address, val,
                                   __acpp_sscp_memory_scope::device);
}

inline unsigned int atomicInc_system(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicInc(address, val,
                                   __acpp_sscp_memory_scope::system);
}

inline unsigned int atomicInc_block(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicInc(address, val,
                                   __acpp_sscp_memory_scope::work_group);
}

inline unsigned int atomicDec(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicDec(address, val,
                                   __acpp_sscp_memory_scope::device);
}

inline unsigned int atomicDec_system(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicDec(address, val,
                                   __acpp_sscp_memory_scope::system);
}

inline unsigned int atomicDec_block(unsigned int *address, unsigned int val) {
  return __pcuda_detail::atomicDec(address, val,
                                   __acpp_sscp_memory_scope::work_group);
}


#endif
