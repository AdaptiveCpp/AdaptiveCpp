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
#include "hipSYCL/sycl/libkernel/sscp/builtins/atomic.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

//TODO: This should go to a common header if this approach will be used in production
inline constexpr int builtin_memory_order(__acpp_sscp_memory_order o) noexcept {
  switch(o){
    case __acpp_sscp_memory_order::relaxed:
      return __ATOMIC_RELAXED;
    case __acpp_sscp_memory_order::acquire:
      return __ATOMIC_ACQUIRE;
    case __acpp_sscp_memory_order::release:
      return __ATOMIC_RELEASE;
    case __acpp_sscp_memory_order::acq_rel:
      return __ATOMIC_ACQ_REL;
    case __acpp_sscp_memory_order::seq_cst:
      return __ATOMIC_SEQ_CST;
  }
  return __ATOMIC_RELAXED;
}

#ifndef __HIP_MEMORY_SCOPE_SINGLETHREAD
 #define __HIP_MEMORY_SCOPE_SINGLETHREAD 1
#endif

#ifndef __HIP_MEMORY_SCOPE_WAVEFRONT
 #define __HIP_MEMORY_SCOPE_WAVEFRONT 2
#endif

#ifndef __HIP_MEMORY_SCOPE_WORKGROUP
 #define __HIP_MEMORY_SCOPE_WORKGROUP 3
#endif

#ifndef __HIP_MEMORY_SCOPE_AGENT
 #define __HIP_MEMORY_SCOPE_AGENT 4
#endif

#ifndef __HIP_MEMORY_SCOPE_SYSTEM
 #define __HIP_MEMORY_SCOPE_SYSTEM 5
#endif

inline constexpr int builtin_memory_scope(__acpp_sscp_memory_scope s) noexcept {
  switch(s) {
    case __acpp_sscp_memory_scope::work_item:
      return __HIP_MEMORY_SCOPE_SINGLETHREAD;
    case __acpp_sscp_memory_scope::sub_group:
      return __HIP_MEMORY_SCOPE_WAVEFRONT;
    case __acpp_sscp_memory_scope::work_group:
      return __HIP_MEMORY_SCOPE_WORKGROUP;
    case __acpp_sscp_memory_scope::device:
      return __HIP_MEMORY_SCOPE_AGENT;
    case __acpp_sscp_memory_scope::system:
      return __HIP_MEMORY_SCOPE_SYSTEM;
  }
}

HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_unsafe_atomic_fetch_add_f32(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f32 *ptr, __acpp_f32 x) {
    return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));(ptr, x);
}

HIPSYCL_SSCP_BUILTIN __acpp_f64 __acpp_sscp_unsafe_atomic_fetch_add_f64(
    __acpp_sscp_address_space as, __acpp_sscp_memory_order order,
    __acpp_sscp_memory_scope scope, __acpp_f64 *ptr, __acpp_f64 x) {
    return __hip_atomic_fetch_add(ptr, x, builtin_memory_order(order),
                                builtin_memory_scope(scope));
}

