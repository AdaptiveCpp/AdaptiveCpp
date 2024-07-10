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
#ifndef HIPSYCL_HIPLIKE_GLUE_CLANG_HPP
#define HIPSYCL_HIPLIKE_GLUE_CLANG_HPP

#include <cassert>

#include "hipSYCL/sycl/libkernel/backend.hpp"

#if (defined(__clang__) && defined(__HIP__)) || (defined(__clang__) && defined(__CUDA__))

#define __sycl_kernel __attribute__((diagnose_if(false,"hipsycl_kernel","warning")))


// We need these calls for configuring CUDA kernel calls - on AMD there's a similar function
// called hipConfigureCall() that we can use instead.
#ifdef __CUDA__
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                void *stream);

#else // compiling for __HIP__

extern "C" hipError_t __hipPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                 size_t sharedMem,
                                                 hipStream_t stream);

#endif // __CUDA__

#ifdef __CUDA__
static inline void __acpp_push_kernel_call(dim3 grid, dim3 block, size_t shared, cudaStream_t stream)
{
  __cudaPushCallConfiguration(grid, block, shared, stream);
}

#define __acpp_launch_integrated_kernel(f, grid, block, shared_mem, stream, \
                                           ...)                                \
  __acpp_push_kernel_call(grid, block, shared_mem,                          \
                             static_cast<CUstream_st *>(stream));              \
  f(__VA_ARGS__);

#else

static inline void __acpp_push_kernel_call(dim3 grid, dim3 block, size_t shared, hipStream_t stream)
{
  hipError_t err = __hipPushCallConfiguration(grid, block, shared, stream);
  assert(err == hipSuccess);
}

#define __acpp_launch_integrated_kernel(f, grid, block, shared_mem, stream, \
                                           ...)                                \
  __acpp_push_kernel_call(grid, block, shared_mem,                          \
                             static_cast<hipStream_t>(stream));                \
  f(__VA_ARGS__);

#endif
  

#else
 #error This file needs a CUDA or HIP compiler and the hipSYCL clang plugin
#endif

#endif // HIPSYCL_HIPLIKE_GLUE_CLANG_HPP
