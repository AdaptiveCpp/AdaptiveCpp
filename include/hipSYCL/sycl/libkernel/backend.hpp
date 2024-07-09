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

#ifndef ACPP_LIBKERNEL_BACKEND_HPP
#define ACPP_LIBKERNEL_BACKEND_HPP

#include "cuda/cuda_backend.hpp"
#include "hip/hip_backend.hpp"

// These need to be included last, since they need to
// know if we are in any device pass of the other backends.
#include "sscp/sscp_backend.hpp"
#include "host/host_backend.hpp"

// define (legacy?) platform identification macros
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
 #define HIPSYCL_PLATFORM_ROCM
 #define HIPSYCL_PLATFORM_HIP
#endif

#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA
 #define HIPSYCL_PLATFORM_CUDA
#endif

#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
 #define HIPSYCL_PLATFORM_SSCP
 #define HIPSYCL_PLATFORM_LLVM
#endif

#ifndef ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif

#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP ||                                 \
    ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                                \
    ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_DEVICE 1
#else
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_DEVICE 0
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
 #define HIPSYCL_PLATFORM_CPU
#endif

#ifdef ACPP_LIBKERNEL_DEVICE_PASS
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS 1
#else
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS 0
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS &&                                        \
    !ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define SYCL_DEVICE_ONLY
 #ifndef __SYCL_DEVICE_ONLY__
  #define __SYCL_DEVICE_ONLY__ 1
 #endif
#endif

#if !defined(ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS)
 #define ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif

#if ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
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
#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA ||                                \
    ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
#define HIPSYCL_HIPLIKE_BUILTIN __device__ HIPSYCL_FORCE_INLINE
#endif

#ifndef __acpp_if_target_host
 #if !ACPP_LIBKERNEL_IS_DEVICE_PASS
  #define __acpp_if_target_host(...) __VA_ARGS__
 #else
  #define __acpp_if_target_host(...)
 #endif
#endif

#ifndef __acpp_if_target_device
 #if ACPP_LIBKERNEL_IS_DEVICE_PASS
  #define __acpp_if_target_device(...) __VA_ARGS__
 #else
  #define __acpp_if_target_device(...)
 #endif
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
 #define __acpp_if_target_cuda(...) __acpp_if_target_device(__VA_ARGS__)
#else
 #define __acpp_if_target_cuda(...)
#endif
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
 #define __acpp_if_target_hip(...) __acpp_if_target_device(__VA_ARGS__)
#else
 #define __acpp_if_target_hip(...)
#endif
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP ||                                     \
    ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
 #define __acpp_if_target_hiplike(...)                                       \
  __acpp_if_target_device(__VA_ARGS__)
#else
 #define __acpp_if_target_hiplike(...)
#endif
#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
 #define __acpp_if_target_sscp(...) __acpp_if_target_device(__VA_ARGS__)
#else
 #define __acpp_if_target_sscp(...)
#endif

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP // Same as: host pass, with SSCP enabled
#define __acpp_backend_switch(host_code, sscp_code, cuda_code, hip_code)       \
  if (__acpp_sscp_is_host) {                                                   \
    host_code;                                                                 \
  } else {                                                                     \
    sscp_code;                                                                 \
  }
#else
#define __acpp_backend_switch(host_code, sscp_code, cuda_code, hip_code)       \
  __acpp_if_target_host(host_code;) __acpp_if_target_cuda(cuda_code;)          \
      __acpp_if_target_hip(hip_code;)
#endif

#define ACPP_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)                              \
  ((ACPP_LIBKERNEL_IS_DEVICE_PASS_##backend) &&                                \
   !ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS)


// Backwards compatibility
#define __hipsycl_if_target_host(...) __acpp_if_target_host(__VA_ARGS__)
#define __hipsycl_if_target_device(...) __acpp_if_target_device(__VA_ARGS__)
#define __hipsycl_if_target_cuda(...) __acpp_if_target_cuda(__VA_ARGS__)
#define __hipsycl_if_target_hip(...) __acpp_if_target_hip(__VA_ARGS__)
#define __hipsycl_if_target_hiplike(...) __acpp_if_target_hiplike(__VA_ARGS__)
#define __hipsycl_if_target_sscp(...) __acpp_if_target_sscp(__VA_ARGS__)

#ifdef __ACPP_ENABLE_CUDA_TARGET__
 #define __HIPSYCL_ENABLE_CUDA_TARGET__ __ACPP_ENABLE_CUDA_TARGET__
#endif
#ifdef __ACPP_ENABLE_HIP_TARGET__
 #define __HIPSYCL_ENABLE_HIP_TARGET__ __ACPP_ENABLE_HIP_TARGET__
#endif
#ifdef __ACPP_ENABLE_OMPHOST_TARGET__
 #define __HIPSYCL_ENABLE_OMPHOST_TARGET__ __ACPP_ENABLE_OMPHOST_TARGET__
#endif
#ifdef __ACPP_ENABLE_LLVM_SSCP_TARGET__
 #define __HIPSYCL_ENABLE_LLVM_SSCP_TARGET__ __ACPP_ENABLE_LLVM_SSCP_TARGET__
#endif
#ifdef __ACPP_CLANG__
 #define __HIPSYCL_CLANG__ __ACPP_CLANG__
#endif
#ifdef __ACPP_ENABLE_CUDA_TARGET__
 #define __HIPSYCL_ENABLE_CUDA_TARGET__ __ACPP_ENABLE_CUDA_TARGET__
#endif
#ifdef __ACPP_USE_ACCELERATED_CPU__
 #define __HIPSYCL_USE_ACCELERATED_CPU__ __ACPP_USE_ACCELERATED_CPU__
#endif

#ifdef ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA
#endif
#ifdef ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP
#endif
#ifdef ACPP_LIBKERNEL_COMPILER_SUPPORTS_HOST
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HOST ACPP_LIBKERNEL_COMPILER_SUPPORTS_HOST
#endif
#ifdef ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SSCP ACPP_LIBKERNEL_COMPILER_SUPPORTS_SSCP
#endif

#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST ACPP_LIBKERNEL_IS_DEVICE_PASS_HOST
#endif
#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA
#endif
#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP
#endif
#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SSCP ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#endif

#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS
 #define HIPSYCL_LIBKERNEL_IS_DEVICE_PASS ACPP_LIBKERNEL_IS_DEVICE_PASS
#endif

#define HIPSYCL_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)                           \
  ACPP_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)

#ifdef ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS                         \
  ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS
#endif


#endif
