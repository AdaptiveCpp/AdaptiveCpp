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
#ifndef HIPSYCL_HIP_TARGET_HPP
#define HIPSYCL_HIP_TARGET_HPP

#if !defined(HIPSYCL_RT_HIP_TARGET_CUDA) &&                                    \
    !defined(HIPSYCL_RT_HIP_TARGET_ROCM) &&                                    \
    !defined(HIPSYCL_RT_HIP_TARGET_HIPCPU)

#define HIPSYCL_RT_HIP_TARGET_ROCM

#endif

#ifdef HIPSYCL_RT_HIP_TARGET_CUDA
#define __HIP_PLATFORM_NVCC__ // Only needed for ROCm <6.0
#define __HIP_PLATFORM_NVIDIA__
#include <hip/hip_runtime.h>
#elif defined(HIPSYCL_RT_HIP_TARGET_ROCM)
#ifndef __HIP_PLATFORM_HCC__ // Only needed for ROCm <6.0
#define __HIP_PLATFORM_HCC__
#endif
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif
#ifndef __HIP_ROCclr__
#define __HIP_ROCclr__ (1)
#endif
#include <hip/hip_runtime.h>
#elif defined(HIPSYCL_RT_HIP_TARGET_HIPCPU)
#include "hipCPU/hip/hip_runtime.h"
#else
#error HIP Backend: No HIP target was specified
#endif

#endif
