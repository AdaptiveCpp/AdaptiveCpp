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

void sub_group_barrier(unsigned flags, int scope) __attribute__((convergent));
void work_group_barrier(unsigned flags, int scope) __attribute__((convergent));
void atomic_work_item_fence(unsigned flags, int order, int scope);

namespace {

// Mapping from memory scope to the cl_mem_fence_flags value clang expects.
unsigned get_cl_fence_flags(__acpp_sscp_memory_scope scope) {
  constexpr unsigned cl_local_mem_fence = 1;  // CLK_LOCAL_MEM_FENCE
  constexpr unsigned cl_global_mem_fence = 2; // CLK_GLOBAL_MEM_FENCE

  switch (scope) {
  case __acpp_sscp_memory_scope::work_item:
    return 0;
  case __acpp_sscp_memory_scope::sub_group:
  case __acpp_sscp_memory_scope::work_group:
    return cl_local_mem_fence;
  case __acpp_sscp_memory_scope::device:
  case __acpp_sscp_memory_scope::system:
  default:
    return cl_local_mem_fence | cl_global_mem_fence;
  }
}

// Mapping from OpenCL-C memory_order value clang expects
int get_cl_memory_order(__acpp_sscp_memory_order o) {
  switch (o) {
  case __acpp_sscp_memory_order::relaxed:
    return 0;
  case __acpp_sscp_memory_order::acquire:
    return 2;
  case __acpp_sscp_memory_order::release:
    return 3;
  case __acpp_sscp_memory_order::acq_rel:
    return 4;
  case __acpp_sscp_memory_order::seq_cst:
    return 5;
  }
  return 4;
}

// Mapping from OpenCL-C memory_scope value clang expects
int get_cl_memory_scope(__acpp_sscp_memory_scope s) {
  switch (s) {
  case __acpp_sscp_memory_scope::work_item:
    return 0;
  case __acpp_sscp_memory_scope::work_group:
    return 1;
  case __acpp_sscp_memory_scope::device:
    return 2;
  case __acpp_sscp_memory_scope::system:
    return 2; // clspv has no wider scope than device
  case __acpp_sscp_memory_scope::sub_group:
    return 4;
  }
  return 1;
}

} // namespace

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope scope,
                               __acpp_sscp_memory_order) {
  unsigned cl_flags = get_cl_fence_flags(scope);
  int cl_scope = get_cl_memory_scope(scope);
  work_group_barrier(cl_flags, cl_scope);
}

HIPSYCL_SSCP_BUILTIN
void __acpp_sscp_memory_fence(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order order) {
  // A fence at work-item scope or with relaxed order orders nothing. Emitting
  // it is also invalid as SPIR-V does not allow relaxed OpMemoryBarrier.
  if (scope == __acpp_sscp_memory_scope::work_item ||
      order == __acpp_sscp_memory_order::relaxed) {
    return;
  }

  unsigned cl_flags = get_cl_fence_flags(scope);
  int cl_order = get_cl_memory_order(order);
  int cl_scope = get_cl_memory_scope(scope);

  atomic_work_item_fence(cl_flags, cl_order, cl_scope);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order) {
  unsigned cl_flags = get_cl_fence_flags(scope);
  int cl_scope = get_cl_memory_scope(scope);
  sub_group_barrier(cl_flags, cl_scope);
}
