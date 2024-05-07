/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"

#ifndef HIPSYCL_SSCP_REDUCTION_BUILTINS_HPP
#define HIPSYCL_SSCP_REDUCTION_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_work_group_reduce_i8(__hipsycl_sscp_algorithm_op op, __hipsycl_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_work_group_reduce_i16(__hipsycl_sscp_algorithm_op op, __hipsycl_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_work_group_reduce_i32(__hipsycl_sscp_algorithm_op op, __hipsycl_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_work_group_reduce_i64(__hipsycl_sscp_algorithm_op op, __hipsycl_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint8 __hipsycl_sscp_work_group_reduce_u8(__hipsycl_sscp_algorithm_op op, __hipsycl_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint16 __hipsycl_sscp_work_group_reduce_u16(__hipsycl_sscp_algorithm_op op, __hipsycl_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint32 __hipsycl_sscp_work_group_reduce_u32(__hipsycl_sscp_algorithm_op op, __hipsycl_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint64 __hipsycl_sscp_work_group_reduce_u64(__hipsycl_sscp_algorithm_op op, __hipsycl_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f16 __hipsycl_sscp_work_group_reduce_f16(__hipsycl_sscp_algorithm_op op, __hipsycl_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f32 __hipsycl_sscp_work_group_reduce_f32(__hipsycl_sscp_algorithm_op op, __hipsycl_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f64 __hipsycl_sscp_work_group_reduce_f64(__hipsycl_sscp_algorithm_op op, __hipsycl_f64 x);



HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int8 __hipsycl_sscp_sub_group_reduce_i8(__hipsycl_sscp_algorithm_op op, __hipsycl_int8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int16 __hipsycl_sscp_sub_group_reduce_i16(__hipsycl_sscp_algorithm_op op, __hipsycl_int16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int32 __hipsycl_sscp_sub_group_reduce_i32(__hipsycl_sscp_algorithm_op op, __hipsycl_int32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_int64 __hipsycl_sscp_sub_group_reduce_i64(__hipsycl_sscp_algorithm_op op, __hipsycl_int64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint8 __hipsycl_sscp_sub_group_reduce_u8(__hipsycl_sscp_algorithm_op op, __hipsycl_uint8 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint16 __hipsycl_sscp_sub_group_reduce_u16(__hipsycl_sscp_algorithm_op op, __hipsycl_uint16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint32 __hipsycl_sscp_sub_group_reduce_u32(__hipsycl_sscp_algorithm_op op, __hipsycl_uint32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_uint64 __hipsycl_sscp_sub_group_reduce_u64(__hipsycl_sscp_algorithm_op op, __hipsycl_uint64 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f16 __hipsycl_sscp_sub_group_reduce_f16(__hipsycl_sscp_algorithm_op op, __hipsycl_f16 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f32 __hipsycl_sscp_sub_group_reduce_f32(__hipsycl_sscp_algorithm_op op, __hipsycl_f32 x);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
__hipsycl_f64 __hipsycl_sscp_sub_group_reduce_f64(__hipsycl_sscp_algorithm_op op, __hipsycl_f64 x);


#endif
