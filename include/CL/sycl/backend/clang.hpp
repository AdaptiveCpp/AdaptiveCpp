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

#ifndef HIPSYCL_BACKEND_CLANG_HPP
#define HIPSYCL_BACKEND_CLANG_HPP

#if (defined(__clang__) && defined(__HIP__)) || (defined(__clang__) && defined(__CUDA__))

#define __sycl_kernel __attribute__((diagnose_if(false,"hipsycl_kernel","warning"))) 


// We need these calls for configuring CUDA kernel calls - on AMD there's a similar function
// called hipConfigureCall() that we can use instead.
#ifdef __CUDA__
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                size_t sharedMem,
                                                void *stream);

extern "C" unsigned __cudaPopCallConfiguration();

#else // compiling for __HIP__

hipError_t hipConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem, hipStream_t stream);

#endif // __CUDA__


static inline void __hipsycl_push_kernel_call(dim3 grid, dim3 block, size_t shared, hipStream_t stream)
{
#ifdef __CUDA__
  __cudaPushCallConfiguration(grid, block, shared, stream);
#else
  hipConfigureCall(grid, block, shared, stream);
#endif
}


#define __hipsycl_launch_kernel(f, grid, block, shared_mem, stream, ...) \
  __hipsycl_push_kernel_call(grid, block, shared_mem, stream); \
  f(__VA_ARGS__);
  

#else
 #error This file should only be included when compiling with clang \
in CUDA or HIP mode with the hipSYCL clang plugin.
#endif

#endif // HIPSYCL_BACKEND_CLANG_HPP
