/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"

extern "C" [[clang::convergent]] void __hipsycl_cbs_barrier();

__attribute__((always_inline)) void
__hipsycl_cpu_mem_fence(__hipsycl_sscp_memory_scope fence_scope,
                           __hipsycl_sscp_memory_order order) {
// FIXME!
}


HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__hipsycl_sscp_work_group_barrier(__hipsycl_sscp_memory_scope fence_scope,
                                  __hipsycl_sscp_memory_order order) {

  // TODO: Correctly take into account memory order for local_barrier
  __hipsycl_cbs_barrier();
  if(fence_scope != __hipsycl_sscp_memory_scope::work_group) {
    __hipsycl_cpu_mem_fence(fence_scope, order);
  }
}


HIPSYCL_SSCP_CONVERGENT_BUILTIN void
__hipsycl_sscp_sub_group_barrier(__hipsycl_sscp_memory_scope fence_scope,
                                 __hipsycl_sscp_memory_order order) {
  
  __hipsycl_cpu_mem_fence(fence_scope, order);
}
