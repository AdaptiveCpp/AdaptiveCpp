/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#ifndef HIPSYCL_SSCP_INTEGER_BUILTINS_HPP
#define HIPSYCL_SSCP_INTEGER_BUILTINS_HPP

#include "builtin_config.hpp"

HIPSYCL_SSCP_BUILTIN __hipsycl_int32 __hipsycl_sscp_mul24_s32(__hipsycl_int32 a, __hipsycl_int32 b);
HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_mul24_u32(__hipsycl_uint32 a, __hipsycl_uint32 b);

HIPSYCL_SSCP_BUILTIN __hipsycl_uint8 __hipsycl_sscp_clz_u8(__hipsycl_uint8); 	
HIPSYCL_SSCP_BUILTIN __hipsycl_uint16 __hipsycl_sscp_clz_u16(__hipsycl_uint16); 
HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_clz_u32(__hipsycl_uint32); 	
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_clz_u64(__hipsycl_uint64); 	

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_popcount_u32(__hipsycl_uint32);
HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_popcount_u64(__hipsycl_uint64);
#endif
