/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef HIPSYCL_BACKEND_HPP
#define HIPSYCL_BACKEND_HPP

#include <cassert>

// Use this macro to detect hipSYCL from SYCL code
#define __HIPSYCL__

// First, make sure the right HIP headers are included
#ifdef __HIPSYCL_TRANSFORM__
// During legacy source-to-source transformation,
// include hipCPU since we treat all SYCL code as host
// code during parsing.
 #include <hipCPU/hip/hip_runtime.h>
#else
// Otherwise include regular HIP headers if compiling
// for GPU, and hipCPU if compiling for CPU
 #if defined(__CUDACC__)
  #define HIPSYCL_PLATFORM_CUDA
  // Silence deprecation warnings in hip which occur for newer CUDA versions
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  // Required until HIP submodule is updated to include
  // https://github.com/ROCm-Developer-Tools/HIP/pull/1497
  #pragma clang diagnostic ignored "-Wsometimes-uninitialized"
  #include "hip/hip_runtime.h"
  #pragma clang diagnostic pop
 #elif defined(__HIP__) || defined(__HCC__)
  #define HIPSYCL_PLATFORM_HCC
  #include <hip/hip_runtime.h>
 #else
  #define HIPSYCL_PLATFORM_CPU
  #include "hipCPU/hip/hip_runtime.h"
 #endif
#endif

// Use this macro to mark functions that should be available
// everywhere - kernels and host.
#define HIPSYCL_UNIVERSAL_TARGET __host__ __device__

// Use this macro to mark function that should be available
// in kernel code.
//
// On nvcc, there are more restrictions for __host__ __device__
// lambdas - do we want to have __device__ as kernel target
// in manual mode?
#define HIPSYCL_KERNEL_TARGET __host__ __device__


#ifdef __HIP_DEVICE_COMPILE__
 // These macros are set if we are currently compiling for GPU.
 #define __HIPSYCL_DEVICE__
 #define SYCL_DEVICE_ONLY
#endif

#if defined(__HIPCPU__) || defined(SYCL_DEVICE_ONLY)
 // Use this to check if __device__ functions are available.
 // This is not the same as SYCL_DEVICE_ONLY, since hipCPU when
 // compiling for host also exposes __device__ functions!
 #define __HIPSYCL_DEVICE_CALLABLE__
#endif

// This macro _must_ be defined when compiling with the hipSYCL
// clang plugin.
#ifdef HIPSYCL_CLANG
 #include "clang.hpp"
#else
 #define __sycl_kernel __global__
 #define __hipsycl_launch_kernel hipLaunchKernelGGL
#endif


#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_CPU)
  #define HIPSYCL_SVM_SUPPORTED
#endif

#if defined(SYCL_DEVICE_ONLY)
 // If HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO is defined,
 // information about the iteration space like thread id, block id
 // block size, grid size will be queried on demand instead of being
 // stored inside the item, nd_item, etc SYCL classes.
 // Depending on the quality of compiler optimizations, this may or may
 // not help to reduce register pressure.
 // TODO: We need benchmarks to check the impact! Code could be simplified
 // if it turns out that we don't need it and compiler optimizations are
 // reliably good enough.
 //
 // HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO is only available on GPU.
 #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
#endif

#ifdef HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
 #if !defined (__HIPSYCL_DEVICE_CALLABLE__)
  // On demand iteration space info is in particular unavailable on CPU
  #error Ondemand iteration space querying cannot be activated, __device__ functions are unavailable!
 #endif
#endif

#ifdef HIPSYCL_PLATFORM_CPU
 // This enables kernel invocation via the OpenMP host kernel
 // execution model (instead of plain HIP/hipCPU).
 // Ultimately, for runtime device selection between host/gpu
 // this should always be active. However, currently it is not
 // not yet possible to use hipCPU at the same time as regular GPU HIP,
 // so we can only enable compiling for host when the platform is
 // pure CPU.
 #define HIPSYCL_ENABLE_HOST_KERNEL_INVOCATION
#endif

namespace hipsycl {
namespace sycl {
namespace detail {


inline void invalid_host_call()
{
  assert(false && "Host execution when compiling for CUDA/HIP is unsupported");
}

template <class T> inline T invalid_host_call_dummy_return(const T& t)
{
  invalid_host_call();
  return t;
}

template <class T> inline T invalid_host_call_dummy_return()
{
  invalid_host_call();
  return T{};
}


}
}
}

#endif // HIPSYCL_BACKEND_HPP
