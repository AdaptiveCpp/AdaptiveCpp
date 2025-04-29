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

#ifndef ACPP_LIBKERNEL_HIP_BACKEND_HPP
#define ACPP_LIBKERNEL_HIP_BACKEND_HPP

#if defined(__HIP__) || defined(__HCC__)
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP 1
// We need to include HIP headers always to have __HIP_DEVICE_COMPILE__
// available below
 #ifdef __ACPP_ENABLE_HIP_TARGET__
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wunused-result"
  #include <hip/hip_runtime.h>
  #pragma clang diagnostic pop
 #endif
#else
 #define ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP 0
#endif

#if defined(__HIP_DEVICE_COMPILE__)
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP 1

 #ifndef ACPP_LIBKERNEL_DEVICE_PASS
  #define ACPP_LIBKERNEL_DEVICE_PASS
 #endif

 // TODO: Are these even needed anymore?
 #define ACPP_UNIVERSAL_TARGET __host__ __device__
 #define ACPP_KERNEL_TARGET __host__ __device__
 #define ACPP_HOST_TARGET __host__

 #define HIPSYCL_ONDEMAND_ITERATION_SPACE_INFO
#else
 #define ACPP_LIBKERNEL_IS_DEVICE_PASS_HIP 0
#endif

#endif
