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
#ifndef HIPSYCL_PSTL_STDPAR_BUILTINS_HPP
#define HIPSYCL_PSTL_STDPAR_BUILTINS_HPP

#include "sycl_glue.hpp"
#include "stdpar_defs.hpp"

inline void __acpp_stdpar_barrier() noexcept {
  auto& rt = hipsycl::stdpar::detail::stdpar_tls_runtime::get();
  int num_ops = rt.get_num_outstanding_operations();
  if(num_ops > 0) {
    HIPSYCL_DEBUG_INFO << "[stdpar] Initializing wait for " << num_ops
                       << " operations" << std::endl;
    rt.get_queue().wait();
    rt.finalize_offloading_batch();
  }
}

// Compiler does not currently support handling invoke instructions for
// these calls, so mark them as noexcept (which should be fine) such
// that call instructions are generated instead.
//
// The compiler detects calls to this function and tries to postpone
// its calls within the control flow for as long as possible.
HIPSYCL_STDPAR_NOINLINE
extern "C" void __acpp_stdpar_optional_barrier() noexcept {
  __acpp_stdpar_barrier();
}



#endif
