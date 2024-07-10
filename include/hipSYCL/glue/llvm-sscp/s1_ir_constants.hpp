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
#ifndef HIPSYCL_S1_IR_CONSTANTS_HPP
#define HIPSYCL_S1_IR_CONSTANTS_HPP

#define HIPSYCL_SSCP_STAGE1_IR_CONST extern "C"

// Stage 1 IR constants
// These variables need to be initialized by the clang plugin.
HIPSYCL_SSCP_STAGE1_IR_CONST unsigned long long
    __acpp_local_sscp_hcf_object_id;

HIPSYCL_SSCP_STAGE1_IR_CONST unsigned long long
    __acpp_local_sscp_hcf_object_size;

HIPSYCL_SSCP_STAGE1_IR_CONST const char __acpp_local_sscp_hcf_content[];

HIPSYCL_SSCP_STAGE1_IR_CONST int  __acpp_sscp_is_host;
HIPSYCL_SSCP_STAGE1_IR_CONST int  __acpp_sscp_is_device;


#endif
