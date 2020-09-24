/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_LIBKERNEL_HIP_BACKEND_HPP
#define HIPSYCL_LIBKERNEL_HIP_BACKEND_HPP


#if defined(__HIP__) || defined(__HCC__)
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP 1
// We need to include HIP headers always to have __HIP_DEVICE_COMPILE__
// available below
 #ifdef __HIPSYCL_ENABLE_HIP_TARGET__
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-result"
  #include <hip/hip_runtime.h>
  #pragma clang diagnostic pop
 #endif
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP 0
#endif

#if defined(__HIP_DEVICE_COMPILE__)
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP 1

 #ifndef HIPSYCL_LIBKERNEL_DEVICE_PASS
  #define HIPSYCL_LIBKERNEL_DEVICE_PASS
 #endif

 // TODO: Are these even needed anymore?
 #define HIPSYCL_UNIVERSAL_TARGET __host__ __device__
 #define HIPSYCL_KERNEL_TARGET __host__ __device__
 #define HIPSYCL_HOST_TARGET __host__

 #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
#else
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP 0
#endif

#endif
