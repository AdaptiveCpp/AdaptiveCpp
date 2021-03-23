/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_LIBKERNEL_SPIRV_BACKEND_HPP
#define HIPSYCL_LIBKERNEL_SPIRV_BACKEND_HPP

#if defined(__HIPSYCL_SPIRV__)
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV 1

 #ifdef __HIPSYCL_ENABLE_SPIRV_TARGET__
  // Needed for SPIR-V headers
  #ifndef SYCL_EXTERNAL
   #define SYCL_EXTERNAL
  #endif

  #include "spirv_ops.hpp"
  #include "spirv_vars.hpp"
  #include "spirv_types.hpp"
 #endif
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV 0
#endif

#if defined(__HIPSYCL_SPIRV__)
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV 1

 #ifndef HIPSYCL_LIBKERNEL_DEVICE_PASS
  #define HIPSYCL_LIBKERNEL_DEVICE_PASS
 #endif

 #define HIPSYCL_UNIVERSAL_TARGET
 #define HIPSYCL_KERNEL_TARGET
 #define HIPSYCL_HOST_TARGET

 #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
#else
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPRIV 0
#endif

#endif
