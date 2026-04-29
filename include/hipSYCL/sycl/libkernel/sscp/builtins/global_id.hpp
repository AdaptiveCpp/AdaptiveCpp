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


// THIS FILE INCLUDES STANDARD LIBRARY HEADERS AND MUST NOT BE 
// USED IN THE DEFINITION OF SSCP BUILTINS!

#ifndef HIPSYCL_SSCP_BUILTINS_GLOBAL_ID_HPP
#define HIPSYCL_SSCP_BUILTINS_GLOBAL_ID_HPP

#include "core.hpp"
#include "core_typed.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/queries.hpp"

#include <stddef.h>

// This is used to implement the optimization in llvm-to-backend to treat
// all queries as fitting into int.
// The implementation is provided by the compiler and does not need to be implemented
// by backends.
HIPSYCL_SSCP_BUILTIN bool
__acpp_sscp_if_global_sizes_fit_in_int();

// Used as backend-specific optimization. Need to manually use
// Itanium mangling so that it also works when the host is on MSVC ABI.
// Intel CPU OpenCL seems to be able to generate better code in some cases
// if kernels only use global invocation id builtins, so we want to use
// that builtin there.
// For other backends/devices, we calculate global ids manually so that
// we can benefit from our global-id-fits-in-int32 JIT optimization.
extern "C" size_t _Z33__spirv_BuiltInGlobalInvocationIdi(int);

#define ACPP_RETURN_SPIRV_GLOBAL_ID_IF_SPIRV_CPU_DEVICE(Dim)                   \
  using namespace hipsycl::sycl::AdaptiveCpp_jit;                              \
  if (__acpp_sscp_jit_reflect_compiler_backend() == compiler_backend::spirv) { \
    if (__acpp_sscp_jit_reflect_target_is_cpu())                               \
      return _Z33__spirv_BuiltInGlobalInvocationIdi(Dim);                      \
  }

inline size_t __acpp_sscp_get_global_id_x() {
  ACPP_RETURN_SPIRV_GLOBAL_ID_IF_SPIRV_CPU_DEVICE(0)
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_global_id_x<int>();
  } else {
    return __acpp_sscp_typed_get_global_id_x<size_t>();
  }
}

inline size_t __acpp_sscp_get_global_id_y() {
  ACPP_RETURN_SPIRV_GLOBAL_ID_IF_SPIRV_CPU_DEVICE(1)
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_global_id_y<int>();
  } else {
    return __acpp_sscp_typed_get_global_id_y<size_t>();
  }
}

inline size_t __acpp_sscp_get_global_id_z() {
  ACPP_RETURN_SPIRV_GLOBAL_ID_IF_SPIRV_CPU_DEVICE(2)
  if(__acpp_sscp_if_global_sizes_fit_in_int()) {
    return __acpp_sscp_typed_get_global_id_z<int>();
  } else {
    return __acpp_sscp_typed_get_global_id_z<size_t>();
  }
}

#undef ACPP_RETURN_SPIRV_GLOBAL_ID_IF_SPIRV_CPU_DEVICE

#endif
