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

#ifndef HIPSYCL_LIBKERNEL_CUDA_BACKEND_HPP
#define HIPSYCL_LIBKERNEL_CUDA_BACKEND_HPP

#if defined(__CUDACC__)
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA 1
 #if defined(__NVCOMPILER)
  #define HIPSYCL_LIBKERNEL_CUDA_NVCXX
 #else 
  #define HIPSYCL_LIBKERNEL_CUDA_CLANG
 #endif

 #ifdef __HIPSYCL_ENABLE_CUDA_TARGET__
  #include <cuda_runtime_api.h>
 #endif

 #ifdef HIPSYCL_LIBKERNEL_CUDA_NVCXX
  #include <nv/target>
 #endif
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0) \
  || defined(HIPSYCL_LIBKERNEL_CUDA_NVCXX)

 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA 1

 #ifndef HIPSYCL_LIBKERNEL_DEVICE_PASS
  #define HIPSYCL_LIBKERNEL_DEVICE_PASS
 #endif

 // TODO: Are these even needed anymore?
 #define HIPSYCL_UNIVERSAL_TARGET __host__ __device__
 #define HIPSYCL_KERNEL_TARGET __host__ __device__
 #define HIPSYCL_HOST_TARGET __host__

 #ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX
  // On-demand iteration space info is not possible in nvc++
  // since it requires being able to have divergent class
  // definitions between host and device passes.
  #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
 #endif
#else
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA 0
#endif

#ifdef HIPSYCL_LIBKERNEL_CUDA_NVCXX
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 1

#define __hipsycl_if_target_host(...)                                          \
  if target (nv::target::is_host) {                                            \
    __VA_ARGS__                                                                \
  }
#define __hipsycl_if_target_device(...)                                        \
  if target (nv::target::is_device) {                                          \
    __VA_ARGS__                                                                \
  }
#else
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif


#endif
