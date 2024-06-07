/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
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

#include "builtin_config.hpp"

#ifndef HIPSYCL_SSCP_SHUFFLE_BUILTINS_HPP
#define HIPSYCL_SSCP_SHUFFLE_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_sub_group_shl_i8(__hipsycl_int8 value,
                                                __hipsycl_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_sub_group_shl_i16(__hipsycl_int16 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_sub_group_shl_i32(__hipsycl_int32 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_sub_group_shl_i64(__hipsycl_int64 value,
                                                  __hipsycl_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_work_group_shl_i8(__hipsycl_int8 value,
                                                __hipsycl_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_work_group_shl_i16(__hipsycl_int16 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_work_group_shl_i32(__hipsycl_int32 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_work_group_shl_i64(__hipsycl_int64 value,
                                                  __hipsycl_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_sub_group_shr_i8(__hipsycl_int8 value,
                                                __hipsycl_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_sub_group_shr_i16(__hipsycl_int16 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_sub_group_shr_i32(__hipsycl_int32 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_sub_group_shr_i64(__hipsycl_int64 value,
                                                  __hipsycl_uint32 delta);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_work_group_shr_i8(__hipsycl_int8 value,
                                                __hipsycl_uint32 delta);
HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_work_group_shr_i16(__hipsycl_int16 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_work_group_shr_i32(__hipsycl_int32 value,
                                                  __hipsycl_uint32 delta);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_work_group_shr_i64(__hipsycl_int64 value,
                                                  __hipsycl_uint32 delta);



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_sub_group_permute_i8(__hipsycl_int8 value,
                                                   __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_sub_group_permute_i16(__hipsycl_int16 value,
                                                     __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_sub_group_permute_i32(__hipsycl_int32 value,
                                                     __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_sub_group_permute_i64(__hipsycl_int64 value,
                                                     __hipsycl_int32 mask);



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_work_group_permute_i8(__hipsycl_int8 value,
                                                   __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_work_group_permute_i16(__hipsycl_int16 value,
                                                     __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_work_group_permute_i32(__hipsycl_int32 value,
                                                     __hipsycl_int32 mask);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_work_group_permute_i64(__hipsycl_int64 value,
                                                     __hipsycl_int32 mask);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_sub_group_select_i8(__hipsycl_int8 value,
                                                   __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_sub_group_select_i16(__hipsycl_int16 value,
                                                     __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_sub_group_select_i32(__hipsycl_int32 value,
                                                     __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_sub_group_select_i64(__hipsycl_int64 value,
                                                     __hipsycl_int32 id);



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_work_group_select_i8(__hipsycl_int8 value,
                                                   __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_work_group_select_i16(__hipsycl_int16 value,
                                                     __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_work_group_select_i32(__hipsycl_int32 value,
                                                     __hipsycl_int32 id);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_work_group_select_i64(__hipsycl_int64 value,
                                                     __hipsycl_int32 id);

#endif
