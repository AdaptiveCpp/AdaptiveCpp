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

#ifndef HIPSYCL_LIBKERNEL_BACKEND_HPP
#define HIPSYCL_LIBKERNEL_BACKEND_HPP

#include "cuda/cuda_backend.hpp"
#include "hip/hip_backend.hpp"
#include "spirv/spirv_backend.hpp"
#include "host/host_backend.hpp"

// define (legacy?) platform identification macros
#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP
 #define HIPSYCL_PLATFORM_ROCM
 #define HIPSYCL_PLATFORM_HIP
#endif

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA
 #define HIPSYCL_PLATFORM_CUDA
#endif

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV
 #define HIPSYCL_PLATFORM_SPIRV
#endif

#ifndef HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP ||                                 \
    HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                                \
    HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_DEVICE 1
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_DEVICE 0
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
 #define HIPSYCL_PLATFORM_CPU
#endif

#ifdef HIPSYCL_LIBKERNEL_DEVICE_PASS
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS 1
#else
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS 0
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS &&                                        \
    !HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define SYCL_DEVICE_ONLY
 #ifndef __SYCL_DEVICE_ONLY__
  #define __SYCL_DEVICE_ONLY__ 1
 #endif
#endif

#if !defined(HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS)
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif

#if HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define __SYCL_SINGLE_SOURCE__ 1
#endif

#ifdef __clang__
 #define HIPSYCL_FORCE_INLINE \
 __attribute__((always_inline)) __attribute__((flatten)) inline
 #define HIPSYCL_LOOP_SPLIT_ND_KERNEL [[clang::annotate("hipsycl_nd_kernel")]]
 #define HIPSYCL_LOOP_SPLIT_ND_KERNEL_LOCAL_SIZE_ARG [[clang::annotate("hipsycl_nd_kernel_local_size_arg")]]
 #define HIPSYCL_LOOP_SPLIT_BARRIER [[clang::annotate("hipsycl_barrier")]]
#else
 #define HIPSYCL_FORCE_INLINE inline
 #define HIPSYCL_LOOP_SPLIT_ND_KERNEL
 #define HIPSYCL_LOOP_SPLIT_BARRIER
 #define HIPSYCL_LOOP_SPLIT_ND_KERNEL_LOCAL_SIZE_ARG
#endif
#define HIPSYCL_BUILTIN HIPSYCL_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE
#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                                \
    HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP
#define HIPSYCL_HIPLIKE_BUILTIN __device__ HIPSYCL_FORCE_INLINE
#endif

#ifndef __hipsycl_if_target_host
 #if !HIPSYCL_LIBKERNEL_IS_DEVICE_PASS
  #define __hipsycl_if_target_host(...) __VA_ARGS__
 #else
  #define __hipsycl_if_target_host(...)
 #endif
#endif

#ifndef __hipsycl_if_target_device
 #if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS
  #define __hipsycl_if_target_device(...) __VA_ARGS__
 #else
  #define __hipsycl_if_target_device(...)
 #endif
#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
 #define __hipsycl_if_target_cuda(...) __hipsycl_if_target_device(__VA_ARGS__)
#else
 #define __hipsycl_if_target_cuda(...)
#endif
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
 #define __hipsycl_if_target_hip(...) __hipsycl_if_target_device(__VA_ARGS__)
#else
 #define __hipsycl_if_target_hip(...)
#endif
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP ||                                    \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
 #define __hipsycl_if_target_hiplike(...)                                       \
  __hipsycl_if_target_device(__VA_ARGS__)
#else
 #define __hipsycl_if_target_hiplike(...)
#endif
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
 #define __hipsycl_if_target_spirv(...) __hipsycl_if_target_device(__VA_ARGS__)
#else
 #define __hipsycl_if_target_spirv(...)
#endif

#define HIPSYCL_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)                           \
  ((HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_##backend) &&                             \
   !HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS)

#endif
