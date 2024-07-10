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
  #include <cuda_fp16.h>
 #endif

 #ifdef HIPSYCL_LIBKERNEL_CUDA_NVCXX
  #include <nv/target>
 #endif
#else
 #define HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA 0
#endif

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ != 0 \
  && !defined(HIPSYCL_SSCP_LIBKERNEL_LIBRARY)) \
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

#define __acpp_if_target_host(...)                                          \
  if target (nv::target::is_host) {                                            \
    __VA_ARGS__                                                                \
  }
#define __acpp_if_target_device(...)                                        \
  if target (nv::target::is_device) {                                          \
    __VA_ARGS__                                                                \
  }
#else
 #define HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS 0
#endif


#endif
