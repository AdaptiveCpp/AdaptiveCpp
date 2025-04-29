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
#ifndef HIPSYCL_HIPLIKE_GLUE_NVCXX_HPP
#define HIPSYCL_HIPLIKE_GLUE_NVCXX_HPP

#include <cassert>

#include "hipSYCL/sycl/libkernel/backend.hpp"

#if !defined(ACPP_LIBKERNEL_CUDA_NVCXX)
 #error "This file needs nvc++"
#endif


#define __sycl_kernel __global__

#define __acpp_launch_integrated_kernel(f, grid, block, shared_mem, stream, \
                                           ...)                                \
  f<<<grid, block, shared_mem, static_cast<CUstream_st*>(stream)>>>(__VA_ARGS__);


#endif // HIPSYCL_HIPLIKE_GLUE_NVCXX_HPP
