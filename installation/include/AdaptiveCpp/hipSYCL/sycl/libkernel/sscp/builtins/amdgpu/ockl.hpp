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
#ifndef HIPSYCL_SSCP_AMDGPU_OCKL_INTERFACE_HPP
#define HIPSYCL_SSCP_AMDGPU_OCKL_INTERFACE_HPP

#include "../builtin_config.hpp"

extern "C" {

#define __amdgpu_private __attribute__((address_space(5)))
#define __amdgpu_local __attribute__((address_space(3)))
#define __amdgpu_global __attribute__((address_space(1)))
#define __amdgpu_constant __attribute__((address_space(4)))

#define SSHARP __amdgpu_constant __acpp_uint32 *
#define TSHARP __amdgpu_constant __acpp_uint32 *

__acpp_uint8 __ockl_ctz_u8(__acpp_uint8);
__acpp_uint16 __ockl_ctz_u16(__acpp_uint16); 	
__acpp_uint32 __ockl_ctz_u32(__acpp_uint32); 	
__acpp_uint64 __ockl_ctz_u64(__acpp_uint64); 	

__acpp_uint8 __ockl_clz_u8(__acpp_uint8);
__acpp_uint16 __ockl_clz_u16(__acpp_uint16); 	
__acpp_uint32 __ockl_clz_u32(__acpp_uint32); 	
__acpp_uint64 __ockl_clz_u64(__acpp_uint64); 	

__acpp_uint32 __ockl_popcount_u32(__acpp_uint32);
__acpp_uint64 __ockl_popcount_u64(__acpp_uint64); 	

__acpp_int32 __ockl_add_sat_i32(__acpp_int32,__acpp_int32);
__acpp_uint32 __ockl_add_sat_u32(__acpp_uint32,__acpp_uint32); 	
__acpp_int64 __ockl_add_sat_i64(__acpp_int64,__acpp_int64); 	
__acpp_uint64 __ockl_add_sat_u64(__acpp_uint64,__acpp_uint64); 	

__acpp_int32 __ockl_sub_sat_i32(__acpp_int32,__acpp_int32);
__acpp_uint32 __ockl_sub_sat_u32(__acpp_uint32,__acpp_uint32); 	
__acpp_int64 __ockl_sub_sat_i64(__acpp_int64,__acpp_int64); 	
__acpp_uint64 __ockl_sub_sat_u64(__acpp_uint64,__acpp_uint64); 	

__acpp_int32 __ockl_mul_hi_i32(__acpp_int32,__acpp_int32);
__acpp_uint32 __ockl_mul_hi_u32(__acpp_uint32,__acpp_uint32); 	
__acpp_int64 __ockl_mul_hi_i64(__acpp_int64,__acpp_int64); 	
__acpp_uint64 __ockl_mul_hi_u64(__acpp_uint64,__acpp_uint64); 	

__acpp_int32 __ockl_mul24_i32(__acpp_int32,__acpp_int32);
__acpp_uint32 __ockl_mul24_u32(__acpp_uint32,__acpp_uint32); 	

__acpp_uint64 __ockl_cyclectr_u64(void);
__acpp_uint64 __ockl_steadyctr_u64(void);

__acpp_uint32 __ockl_activelane_u32(void);

__acpp_native_f16 __ockl_wfred_add_f16(__acpp_native_f16 x);
__acpp_f32 __ockl_wfred_add_f32(__acpp_f32 x); 	
__acpp_f64 __ockl_wfred_add_f64(__acpp_f64 x); 	
__acpp_int32 __ockl_wfred_add_i32(__acpp_int32 x); 	
__acpp_int64 __ockl_wfred_add_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_add_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_add_u64(__acpp_uint64 x);
__acpp_int32 __ockl_wfred_and_i32(__acpp_int32 x); 	
__acpp_int64 __ockl_wfred_and_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_and_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_and_u64(__acpp_uint64 x); 	
__acpp_native_f16 __ockl_wfred_max_f16(__acpp_native_f16 x);
__acpp_f32 __ockl_wfred_max_f32(__acpp_f32 x); 	
__acpp_f64 __ockl_wfred_max_f64(__acpp_f64 x); 	
__acpp_int32 __ockl_wfred_max_i32(__acpp_int32 x); 	
__acpp_int64 __ockl_wfred_max_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_max_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_max_u64(__acpp_uint64 x); 	
__acpp_native_f16 __ockl_wfred_min_f16(__acpp_native_f16 x);
__acpp_f32 __ockl_wfred_min_f32(__acpp_f32 x); 	
__acpp_f64 __ockl_wfred_min_f64(__acpp_f64 x); 	
__acpp_int32 __ockl_wfred_min_i32(__acpp_int32 x); 	
__acpp_int64 __ockl_wfred_min_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_min_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_min_u64(__acpp_uint64 x); 	
__acpp_int32 __ockl_wfred_or_i32(__acpp_int32 x);
__acpp_int64 __ockl_wfred_or_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_or_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_or_u64(__acpp_uint64 x); 	
__acpp_int32 __ockl_wfred_xor_i32(__acpp_int32 x);
__acpp_int64 __ockl_wfred_xor_i64(__acpp_int64 x); 	
__acpp_uint32 __ockl_wfred_xor_u32(__acpp_uint32 x); 	
__acpp_uint64 __ockl_wfred_xor_u64(__acpp_uint64 x); 	
__acpp_native_f16 __ockl_wfscan_add_f16(__acpp_native_f16 x, bool inclusive);
__acpp_f32 __ockl_wfscan_add_f32(__acpp_f32 x, bool inclusive); 	
__acpp_f64 __ockl_wfscan_add_f64(__acpp_f64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_add_i32(__acpp_int32 x, bool inclusive); 	
__acpp_int64 __ockl_wfscan_add_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_add_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_add_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_and_i32(__acpp_int32 x, bool inclusive);
__acpp_int64 __ockl_wfscan_and_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_and_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_and_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_native_f16 __ockl_wfscan_max_f16(__acpp_native_f16 x, bool inclusive);
__acpp_f32 __ockl_wfscan_max_f32(__acpp_f32 x, bool inclusive); 	
__acpp_f64 __ockl_wfscan_max_f64(__acpp_f64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_max_i32(__acpp_int32 x, bool inclusive); 	
__acpp_int64 __ockl_wfscan_max_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_max_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_max_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_native_f16 __ockl_wfscan_min_f16(__acpp_native_f16 x, bool inclusive);
__acpp_f32 __ockl_wfscan_min_f32(__acpp_f32 x, bool inclusive); 	
__acpp_f64 __ockl_wfscan_min_f64(__acpp_f64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_min_i32(__acpp_int32 x, bool inclusive); 	
__acpp_int64 __ockl_wfscan_min_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_min_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_min_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_or_i32(__acpp_int32 x, bool inclusive);
__acpp_int64 __ockl_wfscan_or_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_or_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_or_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_int32 __ockl_wfscan_xor_i32(__acpp_int32 x, bool inclusive);
__acpp_int64 __ockl_wfscan_xor_i64(__acpp_int64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfscan_xor_u32(__acpp_uint32 x, bool inclusive); 	
__acpp_uint64 __ockl_wfscan_xor_u64(__acpp_uint64 x, bool inclusive); 	
__acpp_uint32 __ockl_wfbcast_u32(__acpp_uint32 x, __acpp_uint32 i);
__acpp_uint64 __ockl_wfbcast_u64(__acpp_uint64 x, __acpp_uint32 i); 	

bool __ockl_wfany_i32(__acpp_int32 e);
bool __ockl_wfall_i32(__acpp_int32 e);
bool __ockl_wfsame_i32(__acpp_int32 e);

__acpp_uint32 __ockl_bfm_u32(__acpp_uint32,__acpp_uint32);
__acpp_int32 __ockl_bfe_i32(__acpp_int32, __acpp_uint32, __acpp_uint32);
__acpp_uint32 __ockl_bfe_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32); 	
__acpp_uint32 __ockl_bitalign_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_bytealign_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_lerp_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_f32 __ockl_max3_f32(__acpp_f32,__acpp_f32,__acpp_f32);
__acpp_native_f16 __ockl_max3_f16(__acpp_native_f16,__acpp_native_f16,__acpp_native_f16); 	
__acpp_int32 __ockl_max3_i32(__acpp_int32,__acpp_int32,__acpp_int32); 	
__acpp_uint32 __ockl_max3_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32); 	
__acpp_f32 __ockl_median3_f32(__acpp_f32,__acpp_f32,__acpp_f32);
__acpp_native_f16 __ockl_median3_f16(__acpp_native_f16,__acpp_native_f16,__acpp_native_f16); 	
__acpp_int32 __ockl_median3_i32(__acpp_int32,__acpp_int32,__acpp_int32); 	
__acpp_uint32 __ockl_median3_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32); 	
__acpp_f32 __ockl_min3_f32(__acpp_f32,__acpp_f32,__acpp_f32);
__acpp_native_f16 __ockl_min3_f16(__acpp_native_f16,__acpp_native_f16,__acpp_native_f16);
__acpp_int32 __ockl_min3_i32(__acpp_int32,__acpp_int32,__acpp_int32);
__acpp_uint32 __ockl_min3_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint64 __ockl_mqsad_u64(__acpp_uint64, __acpp_uint32, __acpp_uint64);
__acpp_uint32 __ockl_pack_u32(__acpp_f32_4);
__acpp_uint64 __ockl_qsad_u64(__acpp_uint64, __acpp_uint32, __acpp_uint64);
__acpp_uint32 __ockl_msad_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_sad_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_sadd_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_sadhi_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32);
__acpp_uint32 __ockl_sadw_u32(__acpp_uint32,__acpp_uint32,__acpp_uint32); 
__acpp_f32 __ockl_unpack0_f32(__acpp_uint32);
__acpp_f32 __ockl_unpack1_f32(__acpp_uint32);
__acpp_f32 __ockl_unpack2_f32(__acpp_uint32);
__acpp_f32 __ockl_unpack3_f32(__acpp_uint32);
 	
__acpp_f32_4 __ockl_image_load_1D(TSHARP i, __acpp_int32 c);
__acpp_f32_4 __ockl_image_load_1Da(TSHARP i, __acpp_int32_2 c);
__acpp_f32_4 __ockl_image_load_1Db(TSHARP i, __acpp_int32 c);
__acpp_f32_4 __ockl_image_load_2D(TSHARP i, __acpp_int32_2 c);
__acpp_f32_4 __ockl_image_load_2Da(TSHARP i, __acpp_int32_4 c);
__acpp_f32 __ockl_image_load_2Dad(TSHARP i, __acpp_int32_4 c);
__acpp_f32 __ockl_image_load_2Dd(TSHARP i, __acpp_int32_2 c);
__acpp_f32_4 __ockl_image_load_3D(TSHARP i, __acpp_int32_4 c);
__acpp_f32_4 __ockl_image_load_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f);
__acpp_f32_4 __ockl_image_load_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f);

__acpp_f32_4 __ockl_image_load_mip_1D(TSHARP i, __acpp_int32 c, __acpp_int32 l);
__acpp_f32_4 __ockl_image_load_mip_1Da(TSHARP i, __acpp_int32_2 c, __acpp_int32 l); 	
__acpp_f32_4 __ockl_image_load_mip_2D(TSHARP i, __acpp_int32_2 c, __acpp_int32 l); 	
__acpp_f32_4 __ockl_image_load_mip_2Da(TSHARP i, __acpp_int32_4 c, __acpp_int32 l); 	
__acpp_f32 __ockl_image_load_mip_2Dad(TSHARP i, __acpp_int32_4 c, __acpp_int32 l); 	
__acpp_f32 __ockl_image_load_mip_2Dd(TSHARP i, __acpp_int32_2 c, __acpp_int32 l); 	
__acpp_f32_4 __ockl_image_load_mip_3D(TSHARP i, __acpp_int32_4 c, __acpp_int32 l); 	
__acpp_f32_4 __ockl_image_load_mip_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_int32 l); 	
__acpp_f32_4 __ockl_image_load_mip_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_int32 l); 	

__acpp_native_f16_4 __ockl_image_loadh_1D(TSHARP i, __acpp_int32 c);
__acpp_native_f16_4 __ockl_image_loadh_1Da(TSHARP i, __acpp_int32_2 c);
__acpp_native_f16_4 __ockl_image_loadh_1Db(TSHARP i, __acpp_int32 c);
__acpp_native_f16_4 __ockl_image_loadh_2D(TSHARP i, __acpp_int32_2 c); 	
__acpp_native_f16_4 __ockl_image_loadh_2Da(TSHARP i, __acpp_int32_4 c); 	
__acpp_native_f16_4 __ockl_image_loadh_3D(TSHARP i, __acpp_int32_4 c); 	
__acpp_native_f16_4 __ockl_image_loadh_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f); 	
__acpp_native_f16_4 __ockl_image_loadh_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_1D(TSHARP i, __acpp_int32 c, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_1Da(TSHARP i, __acpp_int32_2 c, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_2D(TSHARP i, __acpp_int32_2 c, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_2Da(TSHARP i, __acpp_int32_4 c, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_3D(TSHARP i, __acpp_int32_4 c, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_int32 l); 	
__acpp_native_f16_4 __ockl_image_loadh_mip_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_int32 l); 	

void __ockl_image_store_1D(TSHARP i, __acpp_int32 c, __acpp_f32_4 p);
void __ockl_image_store_1Da(TSHARP i, __acpp_int32_2 c, __acpp_f32_4 p); 	
void __ockl_image_store_1Db(TSHARP i, __acpp_int32 c, __acpp_f32_4 p); 	
void __ockl_image_store_2D(TSHARP i, __acpp_int32_2 c, __acpp_f32_4 p); 	
void __ockl_image_store_2Da(TSHARP i, __acpp_int32_4 c, __acpp_f32_4 p); 	
void __ockl_image_store_2Dad(TSHARP i, __acpp_int32_4 c, __acpp_f32 p); 	
void __ockl_image_store_2Dd(TSHARP i, __acpp_int32_2 c, __acpp_f32 p); 	
void __ockl_image_store_3D(TSHARP i, __acpp_int32_4 c, __acpp_f32_4 p); 	
void __ockl_image_store_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_f32_4 p); 	
void __ockl_image_store_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_f32_4 p); 	
void __ockl_image_store_lod_1D(TSHARP i, __acpp_int32 c, __acpp_int32 l, __acpp_f32_4 p);

void __ockl_image_store_lod_1Da(TSHARP i, __acpp_int32_2 c, __acpp_int32 l, __acpp_f32_4 p); 	
void __ockl_image_store_lod_2D(TSHARP i, __acpp_int32_2 c, __acpp_int32 l, __acpp_f32_4 p); 	
void __ockl_image_store_lod_2Da(TSHARP i, __acpp_int32_4 c, __acpp_int32 l, __acpp_f32_4 p); 	
void __ockl_image_store_lod_2Dad(TSHARP i, __acpp_int32_4 c, __acpp_int32 l, __acpp_f32 p); 	
void __ockl_image_store_lod_2Dd(TSHARP i, __acpp_int32_2 c, __acpp_int32 l, __acpp_f32 p); 	
void __ockl_image_store_lod_3D(TSHARP i, __acpp_int32_4 c, __acpp_int32 l, __acpp_f32_4 p); 	
void __ockl_image_store_lod_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_int32 l, __acpp_f32_4 p); 	
void __ockl_image_store_lod_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_int32 l, __acpp_f32_4 p); 	

void __ockl_image_storeh_1D(TSHARP i, __acpp_int32 c, __acpp_native_f16_4 p);
void __ockl_image_storeh_1Da(TSHARP i, __acpp_int32_2 c, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_1Db(TSHARP i, __acpp_int32 c, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_2D(TSHARP i, __acpp_int32_2 c, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_2Da(TSHARP i, __acpp_int32_4 c, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_3D(TSHARP i, __acpp_int32_4 c, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_native_f16_4 p); 	

void __ockl_image_storeh_lod_1D(TSHARP i, __acpp_int32 c, __acpp_int32 l, __acpp_native_f16_4 p);
void __ockl_image_storeh_lod_1Da(TSHARP i, __acpp_int32_2 c, __acpp_int32 l, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_lod_2D(TSHARP i, __acpp_int32_2 c, __acpp_int32 l, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_lod_2Da(TSHARP i, __acpp_int32_4 c, __acpp_int32 l, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_lod_3D(TSHARP i, __acpp_int32_4 c, __acpp_int32 l, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_lod_CM(TSHARP i, __acpp_int32_2 c, __acpp_int32 f, __acpp_int32 l, __acpp_native_f16_4 p); 	
void __ockl_image_storeh_lod_CMa(TSHARP i, __acpp_int32_4 c, __acpp_int32 f, __acpp_int32 l, __acpp_native_f16_4 p); 	

__acpp_f32_4 __ockl_image_sample_1D(TSHARP i, SSHARP s, __acpp_f32 c);
__acpp_f32_4 __ockl_image_sample_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_f32_4 __ockl_image_sample_2D(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_f32_4 __ockl_image_sample_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_f32 __ockl_image_sample_2Dad(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_f32 __ockl_image_sample_2Dd(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_f32_4 __ockl_image_sample_3D(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_f32_4 __ockl_image_sample_CM(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_f32_4 __ockl_image_sample_CMa(TSHARP i, SSHARP s, __acpp_f32_4 c); 	

__acpp_f32_4 __ockl_image_sample_grad_1D(TSHARP i, SSHARP s, __acpp_f32 c, __acpp_f32 dx, __acpp_f32 dy);
__acpp_f32_4 __ockl_image_sample_grad_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 dx, __acpp_f32 dy); 	
__acpp_f32_4 __ockl_image_sample_grad_2D(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_f32_4 __ockl_image_sample_grad_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_f32 __ockl_image_sample_grad_2Dad(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_f32 __ockl_image_sample_grad_2Dd(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_f32_4 __ockl_image_sample_grad_3D(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32_4 dx, __acpp_f32_4 dy); 	

__acpp_f32_4 __ockl_image_sample_lod_1D(TSHARP i, SSHARP s, __acpp_f32 c, __acpp_f32 l);
__acpp_f32_4 __ockl_image_sample_lod_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 l); 	
__acpp_f32_4 __ockl_image_sample_lod_2D(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 l); 	
__acpp_f32_4 __ockl_image_sample_lod_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_f32 __ockl_image_sample_lod_2Dad(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_f32 __ockl_image_sample_lod_2Dd(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 l); 	
__acpp_f32_4 __ockl_image_sample_lod_3D(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_f32_4 __ockl_image_sample_lod_CM(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_f32_4 __ockl_image_sample_lod_CMa(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	

__acpp_native_f16_4 __ockl_image_sampleh_1D(TSHARP i, SSHARP s, __acpp_f32 c);
__acpp_native_f16_4 __ockl_image_sampleh_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_native_f16_4 __ockl_image_sampleh_2D(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_native_f16_4 __ockl_image_sampleh_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_native_f16_4 __ockl_image_sampleh_3D(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_native_f16_4 __ockl_image_sampleh_CM(TSHARP i, SSHARP s, __acpp_f32_4 c); 	
__acpp_native_f16_4 __ockl_image_sampleh_CMa(TSHARP i, SSHARP s, __acpp_f32_4 c); 	

__acpp_native_f16_4 __ockl_image_sampleh_grad_1D(TSHARP i, SSHARP s, __acpp_f32 c, __acpp_f32 dx, __acpp_f32 dy);
__acpp_native_f16_4 __ockl_image_sampleh_grad_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 dx, __acpp_f32 dy); 	
__acpp_native_f16_4 __ockl_image_sampleh_grad_2D(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_native_f16_4 __ockl_image_sampleh_grad_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32_2 dx, __acpp_f32_2 dy); 	
__acpp_native_f16_4 __ockl_image_sampleh_grad_3D(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32_4 dx, __acpp_f32_4 dy); 	

__acpp_native_f16_4 __ockl_image_sampleh_lod_1D(TSHARP i, SSHARP s, __acpp_f32 c, __acpp_f32 l);
__acpp_native_f16_4 __ockl_image_sampleh_lod_1Da(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 l); 	
__acpp_native_f16_4 __ockl_image_sampleh_lod_2D(TSHARP i, SSHARP s, __acpp_f32_2 c, __acpp_f32 l); 	
__acpp_native_f16_4 __ockl_image_sampleh_lod_2Da(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_native_f16_4 __ockl_image_sampleh_lod_3D(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_native_f16_4 __ockl_image_sampleh_lod_CM(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	
__acpp_native_f16_4 __ockl_image_sampleh_lod_CMa(TSHARP i, SSHARP s, __acpp_f32_4 c, __acpp_f32 l); 	

__acpp_f32_4 __ockl_image_gather4r_2D(TSHARP i, SSHARP s, __acpp_f32_2 c);
__acpp_f32_4 __ockl_image_gather4g_2D(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_f32_4 __ockl_image_gather4b_2D(TSHARP i, SSHARP s, __acpp_f32_2 c); 	
__acpp_f32_4 __ockl_image_gather4a_2D(TSHARP i, SSHARP s, __acpp_f32_2 c); 	

__acpp_int32 __ockl_image_array_size_1Da(TSHARP i);
__acpp_int32 __ockl_image_array_size_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_array_size_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_array_size_CMa(TSHARP i); 	

__acpp_int32 __ockl_image_channel_data_type_1D(TSHARP i);
__acpp_int32 __ockl_image_channel_data_type_1Da(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_1Db(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_2D(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_2Dd(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_3D(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_CM(TSHARP i); 	
__acpp_int32 __ockl_image_channel_data_type_CMa(TSHARP i); 	

__acpp_int32 __ockl_image_channel_order_1D(TSHARP i);
__acpp_int32 __ockl_image_channel_order_1Da(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_1Db(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_2D(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_2Dd(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_3D(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_CM(TSHARP i); 	
__acpp_int32 __ockl_image_channel_order_CMa(TSHARP i); 	

__acpp_int32 __ockl_image_depth_3D(TSHARP i);

__acpp_int32 __ockl_image_height_2D(TSHARP i);
__acpp_int32 __ockl_image_height_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_height_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_height_2Dd(TSHARP i); 	
__acpp_int32 __ockl_image_height_3D(TSHARP i); 	
__acpp_int32 __ockl_image_height_CM(TSHARP i); 	
__acpp_int32 __ockl_image_height_CMa(TSHARP i); 	

__acpp_int32 __ockl_image_num_mip_levels_1D(TSHARP i);
__acpp_int32 __ockl_image_num_mip_levels_1Da(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_2D(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_2Dd(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_3D(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_CM(TSHARP i); 	
__acpp_int32 __ockl_image_num_mip_levels_CMa(TSHARP i); 	

__acpp_int32 __ockl_image_width_1D(TSHARP i);
__acpp_int32 __ockl_image_width_1Da(TSHARP i); 	
__acpp_int32 __ockl_image_width_1Db(TSHARP i); 	
__acpp_int32 __ockl_image_width_2D(TSHARP i); 	
__acpp_int32 __ockl_image_width_2Da(TSHARP i); 	
__acpp_int32 __ockl_image_width_2Dad(TSHARP i); 	
__acpp_int32 __ockl_image_width_2Dd(TSHARP i); 	
__acpp_int32 __ockl_image_width_3D(TSHARP i); 	
__acpp_int32 __ockl_image_width_CM(TSHARP i); 	
__acpp_int32 __ockl_image_width_CMa(TSHARP i); 	

__acpp_uint64 __ockl_get_global_offset(__acpp_uint32);
__acpp_uint64 __ockl_get_global_id(__acpp_uint32);
__acpp_uint64 __ockl_get_local_id(__acpp_uint32);
__acpp_uint64 __ockl_get_group_id(__acpp_uint32);
__acpp_uint64 __ockl_get_global_size(__acpp_uint32);
__acpp_uint64 __ockl_get_local_size(__acpp_uint32);
__acpp_uint64 __ockl_get_num_groups(__acpp_uint32);
__acpp_uint32 __ockl_get_work_dim(void);
__acpp_uint64 __ockl_get_enqueued_local_size(__acpp_uint32);
__acpp_uint64 __ockl_get_global_linear_id(void);
__acpp_uint64 __ockl_get_local_linear_id(void);

bool __ockl_is_local_addr(const void *);
bool __ockl_is_private_addr(const void *);
__amdgpu_global void * __ockl_to_global(void *);
__amdgpu_local void * __ockl_to_local(void *);
__amdgpu_private void * __ockl_to_private(void *);

}

#endif
