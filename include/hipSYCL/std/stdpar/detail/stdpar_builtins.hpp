/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#ifndef HIPSYCL_PSTL_STDPAR_BUILTINS_HPP
#define HIPSYCL_PSTL_STDPAR_BUILTINS_HPP

#include "sycl_glue.hpp"
#include "stdpar_defs.hpp"

inline void __hipsycl_stdpar_barrier() noexcept {
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
extern "C" void __hipsycl_stdpar_optional_barrier() noexcept {
  __hipsycl_stdpar_barrier();
}



#endif
