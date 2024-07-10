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
