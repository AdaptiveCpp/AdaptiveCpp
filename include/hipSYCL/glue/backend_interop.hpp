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


#ifndef HIPSYCL_GLUE_BACKEND_INTEROP_HPP
#define HIPSYCL_GLUE_BACKEND_INTEROP_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/multi_queue_executor.hpp"

#include "hipSYCL/sycl/backend.hpp"

namespace hipsycl {
namespace glue {

template <sycl::backend b> struct backend_interop {
  // Specializations should define for interop with a sycl type T:
  //
  // using native_T_type = <native-backend-type>
  // static native_T_type get_native_T(const T&)
  // T make_T(const native_T_type&, <potentially additional args>)
  //
  // For interop_handle, the following is required:
  // native_queue_type get_native_queue(rt::backend_kernel_launcher*)
  // native_queue_type get_native_queue(rt::device_id, rt::backend_executor*)
  // 
  // In any case, the following should be defined:
  // static constexpr bool can_make_T = <whether make_T exists>
  // static constexpr bool can_extract_native_T = <whether get_native_T exists>
};

}
} // namespace hipsycl

#include "cuda/cuda_interop.hpp"
#include "hip/hip_interop.hpp"
#include "ze/ze_interop.hpp"
#include "omp/omp_interop.hpp"

#endif
