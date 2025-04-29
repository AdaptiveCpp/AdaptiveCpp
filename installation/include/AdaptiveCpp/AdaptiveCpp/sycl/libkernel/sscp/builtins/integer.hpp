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
#ifndef HIPSYCL_SSCP_INTEGER_BUILTINS_HPP
#define HIPSYCL_SSCP_INTEGER_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_BUILTIN __acpp_int32 __acpp_sscp_mul24_s32(__acpp_int32 a, __acpp_int32 b);
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_mul24_u32(__acpp_uint32 a, __acpp_uint32 b);

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_ctz_u8(__acpp_uint8); 	
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_ctz_u16(__acpp_uint16); 
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_ctz_u32(__acpp_uint32); 	
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_ctz_u64(__acpp_uint64); 	

HIPSYCL_SSCP_BUILTIN __acpp_uint8 __acpp_sscp_clz_u8(__acpp_uint8); 	
HIPSYCL_SSCP_BUILTIN __acpp_uint16 __acpp_sscp_clz_u16(__acpp_uint16); 
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_clz_u32(__acpp_uint32); 	
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_clz_u64(__acpp_uint64); 	

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_popcount_u32(__acpp_uint32);
HIPSYCL_SSCP_BUILTIN __acpp_uint64 __acpp_sscp_popcount_u64(__acpp_uint64);
#endif
