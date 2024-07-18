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

#ifndef ACPP_LIBKERNEL_CUDA_BACKEND_HPP
#define ACPP_LIBKERNEL_CUDA_BACKEND_HPP

#if defined(__CUDACC__)
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA 1
 #if defined(__NVCOMPILER)
  #define ACPP_LIBKERNEL_CUDA_NVCXX
 #else 
  #define ACPP_LIBKERNEL_CUDA_CLANG
 #endif

 #ifdef __ACPP_ENABLE_CUDA_TARGET__
  #include <cuda_runtime_api.h>
  #include <cuda_fp16.h>
 #endif

 #ifdef ACPP_LIBKERNEL_CUDA_NVCXX
  #include <nv/target>
 #endif
#else
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0 \
  && !defined(HIPSYCL_SSCP_LIBKERNEL_LIBRARY)) \
  || defined(ACPP_LIBKERNEL_CUDA_NVCXX)

 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA 1

 #ifndef ACPP_LIBKERNEL_DEVICE_PASS
  #define ACPP_LIBKERNEL_DEVICE_PASS
 #endif

 // TODO: Are these even needed anymore?
 #define ACPP_UNIVERSAL_TARGET __host__ __device__
 #define ACPP_KERNEL_TARGET __host__ __device__
 #define ACPP_HOST_TARGET __host__

 #ifndef ACPP_LIBKERNEL_CUDA_NVCXX
  // On-demand iteration space info is not possible in nvc++
  // since it requires being able to have divergent class
  // definitions between host and device passes.
  #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
 #endif
#else
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_CUDA 0
#endif

#ifdef ACPP_LIBKERNEL_CUDA_NVCXX
 #define ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 1

#define __acpp_if_target_host(...)                                          \
  if target (nv::target::is_host) {                                            \
    __VA_ARGS__                                                                \
  }
#define __acpp_if_target_device(...)                                        \
  if target (nv::target::is_device) {                                          \
    __VA_ARGS__                                                                \
  }
#else
 #define ACPP_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif


#endif
