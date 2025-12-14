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

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/amdgpu_general.hpp"

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_read_first_lane(__acpp_int32 input){
    return __builtin_amdgcn_readfirstlane(input);
};
HIPSYCL_SSCP_BUILTIN __acpp_f32 __acpp_sscp_fractf(__acpp_f32 input){
    return __builtin_amdgcn_fractf(input);
}

