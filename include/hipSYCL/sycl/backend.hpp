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
#ifndef HIPSYCL_SYCL_BACKEND_HPP
#define HIPSYCL_SYCL_BACKEND_HPP

#include "hipSYCL/runtime/device_id.hpp"

#include "libkernel/backend.hpp"

namespace hipsycl {
namespace sycl {

using backend = hipsycl::rt::backend_id;

#if defined(ACPP_LIBKERNEL_COMPILER_SUPPORTS_HOST) || defined(__ACPP_ENABLE_OMPHOST_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_OMPHOST
 #define SYCL_EXT_ACPP_BACKEND_OMPHOST
 #define SYCL_EXT_ACPP_BACKEND_OMP
#endif
// In explicit multipass mode, HIPSYCL_PLATFORM_* is not defined in the host
// pass. We therefore consider a backend as enabled either
// * if we have CUDA/HIP language extensions, e.g. HIPSYCL_PLATFORM_* is defined
// * or we are generating kernels for the backend, i.e.
//   __ACPP_ENABLE_*_TARGET__ is defined.
// Note: This might not be entirely correct. Those macros should be defined
// if a backend is available for interop, which would correspond to whether
// the runtime has been compiled with support for a backend.
#if defined(ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP) || defined(__ACPP_ENABLE_HIP_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_HIP
 #define SYCL_EXT_ACPP_BACKEND_HIP
#endif

#if defined(ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA) || defined(__ACPP_ENABLE_CUDA_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_CUDA
 #define SYCL_EXT_ACPP_BACKEND_CUDA
#endif

#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
 #define SYCL_EXT_ACPP_BACKEND_CUDA
 #define SYCL_EXT_ACPP_BACKEND_HIP
 #define SYCL_EXT_ACPP_BACKEND_OCL
 #define SYCL_EXT_ACPP_BACKEND_LEVEL_ZERO
 #define SYCL_EXT_ACPP_BACKEND_OMP
#endif

}
} // namespace hipsycl

#endif
