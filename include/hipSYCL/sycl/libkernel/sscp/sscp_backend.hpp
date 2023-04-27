/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_LIBKERNEL_SSCP_BACKEND_HPP
#define HIPSYCL_LIBKERNEL_SSCP_BACKEND_HPP

// Expose SSCP only when we are targeting it, and we
// are not in a device pass for an SMCP backend. Otherwise,
// it gets  difficult to distinguish the proper SSCP device code
// path since SSCP outlining happens only in the host pass.
#if defined(__HIPSYCL_ENABLE_LLVM_SSCP_TARGET__) &&                            \
    !defined(HIPSYCL_LIBKERNEL_DEVICE_PASS)

 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SSCP 1
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP 1

 #ifndef HIPSYCL_LIBKERNEL_DEVICE_PASS
  #define HIPSYCL_LIBKERNEL_DEVICE_PASS
 #endif

 #ifdef HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
  #undef HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #endif
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 1

 #define __hipsycl_if_target_host(...)                                         \
  if (__hipsycl_sscp_is_host) {                                                \
    __VA_ARGS__                                                                \
  }
 #define __hipsycl_if_target_device(...)                                       \
  if (__hipsycl_sscp_is_device) {                                              \
    __VA_ARGS__                                                                \
  }


 // TODO: Do we need those still?
 #ifndef HIPSYCL_UNIVERSAL_TARGET
  #define HIPSYCL_UNIVERSAL_TARGET
 #endif

 #ifndef HIPSYCL_KERNEL_TARGET
  #define HIPSYCL_KERNEL_TARGET
 #endif

 #ifndef HIPSYCL_HOST_TARGET
  #define HIPSYCL_HOST_TARGET
 #endif

 #include "hipSYCL/glue/llvm-sscp/ir_constants.hpp"
 #include "builtins/core.hpp"
 
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SSCP 0
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP 0
#endif



#endif
