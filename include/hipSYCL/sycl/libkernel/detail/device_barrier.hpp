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
#ifndef HIPSYCL_DEVICE_BARRIER_HPP
#define HIPSYCL_DEVICE_BARRIER_HPP

#include <cassert>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {


#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
inline void sscp_barrier(access::fence_space space) {
  if(space == access::fence_space::local_space) {
    __acpp_sscp_work_group_barrier(memory_scope::work_group,
                                      memory_order::seq_cst);
  } else {
    __acpp_sscp_work_group_barrier(memory_scope::device,
                                      memory_order::seq_cst);
  }
}

#endif

ACPP_KERNEL_TARGET
inline void local_device_barrier(
    access::fence_space space = access::fence_space::global_and_local) {

  __acpp_backend_switch(
      assert(false && "device barrier called on CPU, this should not happen"), 
      sscp_barrier(space),
      __syncthreads(),
      __syncthreads());
}

}
}
}

#endif
