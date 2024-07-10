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

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_work_group_barrier(__acpp_sscp_memory_scope fence_scope,
                               __acpp_sscp_memory_order) {

  if(fence_scope == hipsycl::sycl::memory_scope::system) {
    __nvvm_membar_sys();
  } else if(fence_scope == hipsycl::sycl::memory_scope::device) {
    __nvvm_membar_gl();
  }
  // syncthreads is already a clang builtin
  __syncthreads();
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__acpp_sscp_sub_group_barrier(__acpp_sscp_memory_scope fence_scope,
                              __acpp_sscp_memory_order) {

  if(fence_scope == hipsycl::sycl::memory_scope::system) {
    __nvvm_membar_sys();
  } else if(fence_scope == hipsycl::sycl::memory_scope::device) {
    __nvvm_membar_gl();
  } else if(fence_scope == hipsycl::sycl::memory_scope::work_group) {
    __nvvm_membar_cta();
  }
  // We cannot call __nvvm_bar_warp_sync(-1) since this builtin
  // is only available for ptx 60 or newer - but at this point,
  // we don't know yet which capabilities we are targeting.
  // TODO: Disable this line if ptx < 60
  asm("bar.warp.sync -1;");
}
