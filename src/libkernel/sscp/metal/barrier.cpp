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
#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"

#include "helpers.hpp"

using namespace hipsycl::sycl::detail::metal_builtins;

HIPSYCL_SSCP_CONVERGENT_BUILTIN void __acpp_sscp_metal_symbol_barrier(const char* s);

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order order)
{
  switch (order) {
    case __acpp_sscp_memory_order::relaxed:
    case __acpp_sscp_memory_order::acquire:
    case __acpp_sscp_memory_order::release:
    case __acpp_sscp_memory_order::acq_rel:
    case __acpp_sscp_memory_order::seq_cst:
      break;
    default:
      __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
      break;
  }

  auto fence = [&]() {
    switch (fence_scope) {
      case __acpp_sscp_memory_scope::work_item:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_none, memory_order_seq_cst, thread_scope_thread)");
        break;
      case __acpp_sscp_memory_scope::sub_group:
      case __acpp_sscp_memory_scope::work_group:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup)");
        break;
      case __acpp_sscp_memory_scope::device:
      case __acpp_sscp_memory_scope::system:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)");
      default:
        __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
        break;
    }
  };

  if (order == __acpp_sscp_memory_order::release
    || order == __acpp_sscp_memory_order::acq_rel
    || order == __acpp_sscp_memory_order::seq_cst
  ) {
    fence();
  }

  switch (fence_scope) {
    case __acpp_sscp_memory_scope::work_item:
      __acpp_sscp_metal_symbol_barrier("threadgroup_barrier(mem_flags::mem_none)");
      break;
    case __acpp_sscp_memory_scope::sub_group:
      __acpp_sscp_metal_symbol_barrier("simdgroup_barrier(mem_flags::mem_threadgroup)");
      break;
    case __acpp_sscp_memory_scope::work_group:
      __acpp_sscp_metal_symbol_barrier("threadgroup_barrier(mem_flags::mem_threadgroup)");
      break;
    case __acpp_sscp_memory_scope::device:
    case __acpp_sscp_memory_scope::system:
      __acpp_sscp_metal_symbol_barrier("threadgroup_barrier(mem_flags::mem_device)");
      break;
    default:
      __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
      break;
  }

  if (order == __acpp_sscp_memory_order::acquire
    || order == __acpp_sscp_memory_order::acq_rel
    || order == __acpp_sscp_memory_order::seq_cst
  ) {
    fence();
  }
}


HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order order)
{
  switch (order) {
    case __acpp_sscp_memory_order::relaxed:
    case __acpp_sscp_memory_order::acquire:
    case __acpp_sscp_memory_order::release:
    case __acpp_sscp_memory_order::acq_rel:
    case __acpp_sscp_memory_order::seq_cst:
      break;
    default:
      __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
      break;
  }

  auto fence = [&]() {
    switch (fence_scope) {
      case __acpp_sscp_memory_scope::work_item:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_none, memory_order_seq_cst, thread_scope_thread)");
        break;
      case __acpp_sscp_memory_scope::sub_group:
      case __acpp_sscp_memory_scope::work_group:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup)");
        break;
      case __acpp_sscp_memory_scope::device:
      case __acpp_sscp_memory_scope::system:
        __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)");
      default:
        __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
        break;
    }
  };

  if (order == __acpp_sscp_memory_order::release
    || order == __acpp_sscp_memory_order::acq_rel
    || order == __acpp_sscp_memory_order::seq_cst
  ) {
    fence();
  }

  switch (fence_scope) {
    case __acpp_sscp_memory_scope::work_item:
      __acpp_sscp_metal_symbol_barrier("simdgroup_barrier(mem_flags::mem_none)");
      break;
    case __acpp_sscp_memory_scope::sub_group:
      __acpp_sscp_metal_symbol_barrier("simdgroup_barrier(mem_flags::mem_threadgroup)");
      break;
    case __acpp_sscp_memory_scope::work_group:
      __acpp_sscp_metal_symbol_barrier("simdgroup_barrier(mem_flags::mem_threadgroup)");
      break;
    case __acpp_sscp_memory_scope::device:
    case __acpp_sscp_memory_scope::system:
      __acpp_sscp_metal_symbol_barrier("simdgroup_barrier(mem_flags::mem_device)");
      break;
    default:
      __acpp_sscp_metal_symbol_barrier("__builtin_trap()");
      break;
  }

  if (order == __acpp_sscp_memory_order::acquire
    || order == __acpp_sscp_memory_order::acq_rel
    || order == __acpp_sscp_memory_order::seq_cst
  ) {
    fence();
  }
}

HIPSYCL_SSCP_BUILTIN
void __acpp_sscp_memory_fence(__acpp_sscp_memory_scope scope,
                              __acpp_sscp_memory_order order)
{
  switch(order) {
    // Metal spec: atomic_thread_fence with memory_order_relaxed has no effect
    case __acpp_sscp_memory_order::relaxed:
      break;
    case __acpp_sscp_memory_order::acquire:
    case __acpp_sscp_memory_order::release:
    case __acpp_sscp_memory_order::acq_rel:
    case __acpp_sscp_memory_order::seq_cst: {
      switch (scope) {
        case __acpp_sscp_memory_scope::work_item:
          __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_none, memory_order_seq_cst, thread_scope_thread)");
          break;
        case __acpp_sscp_memory_scope::sub_group:
          __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_simdgroup)");
          break;
        case __acpp_sscp_memory_scope::work_group:
          __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_threadgroup)");
          break;
        case __acpp_sscp_memory_scope::device:
          __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)");
          break;
        case __acpp_sscp_memory_scope::system:
          __acpp_sscp_metal_symbol_barrier("atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device)");
          break;
        default:
          __builtin_trap();
          break;
      }
      break;
    }
    default: __builtin_trap();
  }
}
