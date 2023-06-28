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

#ifndef HIPSYCL_SSCP_MUSA_LIBDEVICE_INTERFACE_HPP
#define HIPSYCL_SSCP_MUSA_LIBDEVICE_INTERFACE_HPP

#include "../builtin_config.hpp"

// atomic operations
extern "C" float __mt_atomicAdd_f32(float *, float);
extern "C" double __mt_atomicAdd_f64(double *, double);
extern "C" __hipsycl_uint32 __mt_atomicInc_u32(__hipsycl_uint32 *__addr,
                                             __hipsycl_uint32 __val);
extern "C" __hipsycl_uint32 __mt_atomicDec_u32(__hipsycl_uint32 *__addr,
                                             __hipsycl_uint32 __val);

// warp functions
// shfl
extern "C" __hipsycl_int32 __mt_shfl_idx_sync_i32(__hipsycl_uint32, __hipsycl_int32, __hipsycl_int32, __hipsycl_int32, bool *, __hipsycl_int32 *);
extern "C" __hipsycl_int32 __mt_shfl_up_sync_i32(__hipsycl_uint32, __hipsycl_int32, __hipsycl_uint32, __hipsycl_int32, bool *, __hipsycl_int32 *);
extern "C" __hipsycl_int32 __mt_shfl_down_sync_i32(__hipsycl_uint32, __hipsycl_int32, __hipsycl_uint32, __hipsycl_int32, bool *, __hipsycl_int32 *);
extern "C" __hipsycl_int32 __mt_shfl_xor_sync_i32(__hipsycl_uint32, __hipsycl_int32, __hipsycl_int32, __hipsycl_int32, bool *, __hipsycl_int32 *);
// vote
extern "C" __hipsycl_int32 __mt_vote_all_sync_i32(__hipsycl_uint32, __hipsycl_int32);
extern "C" __hipsycl_int32 __mt_vote_any_sync_i32(__hipsycl_uint32, __hipsycl_int32);
extern "C" __hipsycl_uint32 __mt_vote_ballot_sync_u32(__hipsycl_uint32, __hipsycl_int32);
// match
extern "C" __hipsycl_uint32 __mt_match_any_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_match_any_sync_u64(__hipsycl_uint32, __hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_match_any_sync_f32(__hipsycl_uint32, float);
extern "C" __hipsycl_uint32 __mt_match_all_sync_u32(__hipsycl_uint32, __hipsycl_uint32, __hipsycl_int32 *);
extern "C" __hipsycl_uint32 __mt_match_all_sync_u64(__hipsycl_uint32, __hipsycl_uint32, __hipsycl_uint32, __hipsycl_int32 *);
extern "C" __hipsycl_uint32 __mt_match_all_sync_f32(__hipsycl_uint32, float, __hipsycl_int32 *);
// reduce
extern "C" __hipsycl_uint32 __mt_reduce_add_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_reduce_min_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_reduce_max_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_int32 __mt_reduce_add_sync_i32(__hipsycl_uint32, __hipsycl_int32);
extern "C" __hipsycl_int32 __mt_reduce_min_sync_i32(__hipsycl_uint32, __hipsycl_int32);
extern "C" __hipsycl_int32 __mt_reduce_max_sync_i32(__hipsycl_uint32, __hipsycl_int32);
extern "C" __hipsycl_uint32 __mt_reduce_and_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_reduce_or_sync_u32(__hipsycl_uint32, __hipsycl_uint32);
extern "C" __hipsycl_uint32 __mt_reduce_xor_sync_u32(__hipsycl_uint32, __hipsycl_uint32);

// Math builtin functions on mtgpu implemented in libdevice.
// Single precision math functions using intrinsics.
extern "C" float __mt_fabs_f32(float __a);
extern "C" float __mt_floor_f32(float __f);
extern "C" float __mt_fma_f32(float __a, float __b, float __c);
extern "C" float __mt_fmax_f32(float __a, float __b);
extern "C" float __mt_fmin_f32(float __a, float __b);
extern "C" float __mt_expf_f32(float __a);
extern "C" float __mt_exp2_f32(float __a);
extern "C" float __mt_log2_f32(float __a);
extern "C" float __mt_sin_f32(float __a);
extern "C" float __mt_cos_f32(float __a);
extern "C" float __mt_atan_f32(float __a);
extern "C" float __mt_sinpi_f32(float __a);
extern "C" float __mt_cospi_f32(float __a);

// Self defined single precision math functions.
extern "C" float __mt_fdim_f32(float __a, float __b);
extern "C" float __mt_fmod_f32(float __a, float __b);
extern "C" float __mt_erf_f32(float __a);
extern "C" float __mt_erfc_f32(float __a);
extern "C" float __mt_expm1_f32(float __a);
extern "C" float __mt_remainder_f32(float __a, float __b);
extern "C" float __mt_atan2_f32(float __a, float __b);
extern "C" float __mt_pow_f32(float __a, float __b);
extern "C" float __mt_frexp_f32(float __a, __hipsycl_int32 *__b);
extern "C" float __mt_ldexp_f32(float __a, __hipsycl_int32 __b);
extern "C" float __mt_scalbn_f32(float __a, __hipsycl_int32 __b);
extern "C" float __mt_lgamma_f32(float x);
extern "C" float __mt_log10_f32(float __a);
extern "C" float __mt_log1p_f32(float __a);
extern "C" float __mt_logb_f32(float __a);
extern "C" float __mt_log_f32(float __a);
extern "C" float __mt_remquo_f32(float __a, float __b, __hipsycl_int32 *__c);

// Self defined single precision relational functions.
extern "C" __hipsycl_int32 __mt_isnan_f32(float __a);
extern "C" __hipsycl_int32 __mt_isinf_f32(float __a);
extern "C" __hipsycl_int32 __mt_isfinite_f32(float __a);

// Double precision math functions.
extern "C" __hipsycl_int32 __mt_ilogb_f64(double __a);
extern "C" double __mt_copysign_f64(double __a, double __b);
extern "C" float  __mt_copysign_f32(float __a, float __b);
extern "C" double __mt_erf_f64(double __a);
extern "C" double __mt_erfc_f64(double __a);
extern "C" double __mt_exp_f64(double __a);
extern "C" double __mt_exp2_f64(double __a);
extern "C" double __mt_expm1_f64(double __a);
extern "C" double __mt_fabs_f64(double __a);
extern "C" double __mt_sin_f64(double __a);
extern "C" double __mt_cos_f64(double __a);
extern "C" double __mt_sinpi_f64(double __a);
extern "C" double __mt_cospi_f64(double __a);
extern "C" double __mt_tan_f64(double __a);
extern "C" double __mt_fdim_f64(double __a, double __b);
extern "C" double __mt_floor_f64(double __a);
extern "C" double __mt_rint_f64(double __a);
extern "C" float __mt_rint_f32(float __a);
extern "C" __hipsycl_int64 __mt_llrint_f32(float __a);
extern "C" double __mt_fma_f64(double __a, double __b, double __c);
extern "C" double __mt_fmax_f64(double __a, double __b);
extern "C" double __mt_fmin_f64(double __a, double __b);
extern "C" double __mt_fmod_f64(double __a, double __b);
extern "C" double __mt_frexp_f64(double __a, __hipsycl_int32 *__b);
extern "C" double __mt_ldexp_f64(double __a, __hipsycl_int32 __b);
extern "C" double __mt_pow_f64(double __a, double __b);
extern "C" double __mt_remainder_f64(double __a, double __b);
extern "C" double __mt_scalbn_f64(double __a, __hipsycl_int32 __b);
extern "C" double __mt_add_f64(double __a, double __b);
extern "C" double __mt_add_rtn_f64(double __a, double __b);
extern "C" double __mt_add_rtp_f64(double __a, double __b);
extern "C" double __mt_add_rte_f64(double __a, double __b);
extern "C" double __mt_add_rtz_f64(double __a, double __b);
extern "C" double __mt_sub_f64(double __a, double __b);
extern "C" double __mt_sub_rtn_f64(double __a, double __b);
extern "C" double __mt_sub_rtp_f64(double __a, double __b);
extern "C" double __mt_sub_rte_f64(double __a, double __b);
extern "C" double __mt_sub_rtz_f64(double __a, double __b);
extern "C" double __mt_mul_f64(double __a, double __b);
extern "C" double __mt_mul_rtn_f64(double __a, double __b);
extern "C" double __mt_mul_rtp_f64(double __a, double __b);
extern "C" double __mt_mul_rte_f64(double __a, double __b);
extern "C" double __mt_mul_rtz_f64(double __a, double __b);
extern "C" double __mt_div_f64(double __a, double __b);
extern "C" double __mt_div_rtn_f64(double __a, double __b);
extern "C" double __mt_div_rtp_f64(double __a, double __b);
extern "C" double __mt_div_rte_f64(double __a, double __b);
extern "C" double __mt_div_rtz_f64(double __a, double __b);
extern "C" double __mt_rem_f64(double __a, double __b);
extern "C" double __mt_eq_f64(double __a, double __b);
extern "C" double __mt_ne_f64(double __a, double __b);
extern "C" double __mt_gt_f64(double __a, double __b);
extern "C" double __mt_lt_f64(double __a, double __b);
extern "C" double __mt_ge_f64(double __a, double __b);
extern "C" double __mt_le_f64(double __a, double __b);
extern "C" double __mt_exp_cuda_f64(double __a);

extern "C" double __mt_log10_f64(double __a);
extern "C" double __mt_log_f64(double __a);
extern "C" double __mt_log2_f64(double __a);
extern "C" double __mt_logb_f64(double __a);
extern "C" double __mt_log1p_f64(double __a);
extern "C" double __mt_modf_f64(double __a ,double *__b);
extern "C" float __mt_exp_f32(float __a);
extern "C" double __mt_asinh_f64(double __a);
extern "C" double __mt_sinh_f64(double __a);
extern "C" double __mt_cosh_f64(double __a);
extern "C" double __mt_tanh_f64(double __a);
extern "C" double __mt_erf_f64(double __a);
extern "C" double __mt_erfc_f64(double __a);
extern "C" double __mt_lgamma_f64(double __a);
extern "C" __hipsycl_int64 __mt_llrint_f64(double __a);

// Self defined double precision relational functions.
extern "C" __hipsycl_int32 __mt_isnan_f64(double __a);
extern "C" __hipsycl_int32 __mt_isinf_f64(double __a);
extern "C" __hipsycl_int32 __mt_isfinite_f64(double __a);
extern "C" __hipsycl_int32 __mt_isnormal_f64(double __a);
extern "C" __hipsycl_int32 __mt_isunordered_f64(double __a, double __b);
extern "C" __hipsycl_int32 __mt_isless_f64(double __a, double __b);
extern "C" __hipsycl_int32 __mt_isgreater_f64(double __a, double __b);
extern "C" __hipsycl_int32 __mt_islessequal_f64(double __a, double __b);
extern "C" __hipsycl_int32 __mt_isgreaterequal_f64(double __a, double __b);
extern "C" __hipsycl_int32 __mt_islessgreater_f64(double __a, double __b);

// Integer math functions using intrinsics.
extern "C" __hipsycl_int64 __mt_abs_i64(__hipsycl_int64 __a);
extern "C" __hipsycl_int32 __mt_abs_i32(__hipsycl_int32 __a);
extern "C" __hipsycl_int64 __mt_llmax(__hipsycl_int64 __a, __hipsycl_int64 __b);
extern "C" __hipsycl_int64 __mt_llmin(__hipsycl_int64 __a, __hipsycl_int64 __b);
extern "C" __hipsycl_int32 __mt_max(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_int32 __mt_min(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_uint64 __mt_ullmax(__hipsycl_uint64 __a,
                                            __hipsycl_uint64 __b);
extern "C" __hipsycl_uint64 __mt_ullmin(__hipsycl_uint64 __a,
                                            __hipsycl_uint64 __b);
extern "C" __hipsycl_uint32 __mt_umax(__hipsycl_uint32 __a, __hipsycl_uint32 __b);
extern "C" __hipsycl_uint32 __mt_umin(__hipsycl_uint32 __a, __hipsycl_uint32 __b);

extern "C" __hipsycl_int32 __mt_mul24(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_int64 __mt_mul64hi(__hipsycl_int64 __a, __hipsycl_int64 __b);
extern "C" __hipsycl_int32 __mt_mulhi(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_uint32 __mt_umul24(__hipsycl_uint32 __a, __hipsycl_uint32 __b);
extern "C" __hipsycl_uint32 __mt_umulhi(__hipsycl_uint32 __a, __hipsycl_uint32 __b);
extern "C" __hipsycl_uint64 __mt_umul64hi(__hipsycl_uint64 __a,
                                              __hipsycl_uint64 __b);
extern "C" __hipsycl_int32 __mt_hadd(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_uint32 __mt_uhadd(__hipsycl_uint32 __a, __hipsycl_uint32 __b);
extern "C" __hipsycl_int32 __mt_sad(__hipsycl_int32 __a, __hipsycl_int32 __b, __hipsycl_int32 __c);
extern "C" __hipsycl_uint32 __mt_usad(__hipsycl_uint32 __a, __hipsycl_uint32 __b,
                                    __hipsycl_uint32 __c);
extern "C" __hipsycl_int32 __mt_rhadd(__hipsycl_int32 __a, __hipsycl_int32 __b);
extern "C" __hipsycl_uint32 __mt_urhadd(__hipsycl_uint32 __a, __hipsycl_uint32 __b);

// Type conversion functions.
extern "C" double __mt_longlong_as_double(__hipsycl_int64);
extern "C" double __mt_ulonglong_as_double(__hipsycl_uint64);
extern "C" __hipsycl_int64 __mt_double_as_longlong(double);
extern "C" __hipsycl_uint64 __mt_double_as_ulonglong(double);
extern "C" __hipsycl_uint32 __mt_float_as_uint(float);
extern "C" float __mt_uint_as_float(__hipsycl_uint32);
extern "C" __hipsycl_int32 __mt_float_as_int(float);
extern "C" float __mt_int_as_float(__hipsycl_int32);

extern "C" float __mt_ui32_to_f32_rp(__hipsycl_uint32);
extern "C" __hipsycl_uint8 __mt_float2uchar_rz_sat(float x);
extern "C" __hipsycl_int8 __mt_float2char_rz_sat(float x);

extern "C" float __mt_fsub_rn_f32(float __a, float __b);
extern "C" float __mt_fmul_rn_f32(float __a, float __b);
extern "C" float __mt_fmaf_rn_f32(float __a, float __b, float __c);

// following delcarations should be removed after mtgpu libdevice supported.
extern "C" float __mt_acos_f32(float __a);
extern "C" double __mt_acos_f64(double __a);
extern "C" float __mt_acosh_f32(float __a);
extern "C" double __mt_acosh_f64(double __a);
extern "C" double __mt_asin_f64(double __a);
extern "C" float __mt_asin_f32(float __a);
extern "C" float __mt_asinh_f32(float __a);
extern "C" double __mt_atan_f64(double __a);
extern "C" double __mt_atan2_f64(double __a, double __b);
extern "C" float __mt_atan2_f32(float __a, float __b);
extern "C" float __mt_tan_f32(float __a);
extern "C" double __mt_atanh_f64(double __a);
extern "C" float __mt_tanh_f32(float __a);
extern "C" float __mt_sinh_f32(float __a);
extern "C" double __mt_cbrt_f64(double __a);
extern "C" float __mt_cbrt_f32(float __a);
extern "C" double __mt_ceil_f64(double __a);
extern "C" float __mt_ceil_f32(float __a);
extern "C" float __mt_cosh_f32(float __a);
extern "C" __hipsycl_int32 __mt_double_to_i32_rn(double __a);
extern "C" float __mt_fast_fdivide_f32(float __a, float __b);
extern "C" __hipsycl_int32 __mt_ffs_i32(__hipsycl_int32 __a);
extern "C" __hipsycl_int32 __mt_ffsll_i64(__hipsycl_int64 __a);
extern "C" __hipsycl_int32 __mt_f32_to_i32_rd(float __a);
extern "C" __hipsycl_int32 __mt_f32_to_i32_ru(float __a);
extern "C" double __mt_hypot_f64(double __a, double __b);
extern "C" float __mt_hypot_f32(float __a, float __b);
extern "C" __hipsycl_int32 __mt_ilogb_f64(double __a);
extern "C" __hipsycl_int32 __mt_ilogb_f32(float __a);
extern "C" __hipsycl_int64 __mt_llround_f64(double __a);
extern "C" __hipsycl_int64 __mt_llround_cuda_f64(double __a);
extern "C" __hipsycl_int64 __mt_llround_f32(float __a);

extern "C" double __mt_modf_f64(double __a, double *__b);
extern "C" float __mt_modf_f32(float __a, float *__b);

extern "C" float __mt_nextafter_f32(float __a, float __b);
extern "C" float __mt_ncdf_f32(float __a);
extern "C" double __mt_remquo_f64(double __a, double __b, __hipsycl_int32 *__c);
extern "C" double __mt_round_f64(double __a);
extern "C" double __mt_round_cuda_f64(double __a);
extern "C" float __mt_round_f32(float __a);
extern "C" float __mt_rsqrt_f32(float __a);
extern "C" float __mt_saturate_f32(float __a);
extern "C" bool __mt_signbit_f64(double __a);
extern "C" bool __mt_signbit_f32(float __a);
extern "C" double __mt_sqrt_f64(double __a);
extern "C" float __mt_sqrt_f32(float __a);
extern "C" float __mt_nan_f32(const __hipsycl_int8 *__a);
extern "C" double __mt_nan_cuda_f64(const __hipsycl_int8 *__a);
extern "C" float __mt_atanh_f32(float __a);
extern "C" double __mt_tgamma_f64(double __a);
extern "C" float __mt_tgamma_f32(float __a);
extern "C" double __mt_trunc_f64(double __a);
extern "C" float __mt_trunc_f32(float __a);

// undocumented ?
extern "C" double __mt_exp10_f64(double);
extern "C" double __mt_nextafter_f64(double __a, double __b);
extern "C" float __mt_powr_f32(float __a, float __b);
extern "C" double __mt_powr_f64(double __a, double __b);
extern "C" float __mt_pown_f32(float x, __hipsycl_int32 y);
extern "C" double __mt_pown_f64(double x, __hipsycl_int32 y);
extern "C" double __mt_rsqrt_f64(double __a);

#endif
