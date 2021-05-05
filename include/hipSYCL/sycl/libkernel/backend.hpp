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
 #define SYCL_DEVICE_ONLY
#endif

#ifdef __clang__
 #define HIPSYCL_FORCE_INLINE \
 __attribute__((always_inline)) __attribute__((flatten)) inline
#else
 #define HIPSYCL_FORCE_INLINE inline
#endif
#define HIPSYCL_BUILTIN HIPSYCL_UNIVERSAL_TARGET HIPSYCL_FORCE_INLINE

#endif
