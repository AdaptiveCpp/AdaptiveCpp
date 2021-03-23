/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_SYCL_BACKEND_HPP
#define HIPSYCL_SYCL_BACKEND_HPP

#include "hipSYCL/runtime/device_id.hpp"

#include "libkernel/backend.hpp"

namespace hipsycl {
namespace sycl {

using backend = hipsycl::rt::backend_id;

#if defined(HIPSYCL_PLATFORM_CPU) && defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_OMPHOST
#endif
// In explicit multipass mode, HIPSYCL_PLATFORM_* is not defined in the host
// pass. We therefore consider a backend as enabled either
// * if we have CUDA/HIP language extensions, e.g. HIPSYCL_PLATFORM_* is defined
// * or we are generating kernels for the backend, i.e.
//   __HIPSYCL_ENABLE_*_TARGET__ is defined.
// Note: This might not be entirely correct. Those macros should be defined
// if a backend is available for interop, which would correspond to whether
// the runtime has been compiled with support for a backend.
#if defined(HIPSYCL_PLATFORM_HIP) || defined(__HIPSYCL_ENABLE_HIP_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_HIP
#endif

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_CUDA
#endif

#if defined(HIPSYCL_PLATFORM_SPIRV) || defined(__HIPSYCL_ENABLE_SPIRV_TARGET__)
 #define SYCL_EXT_HIPSYCL_BACKEND_SPIRV
#endif

}
} // namespace hipsycl

#endif
