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
#ifndef HIPSYCL_SSCP_AMDGPU_OCLC_INTERFACE_HPP
#define HIPSYCL_SSCP_AMDGPU_OCLC_INTERFACE_HPP

#include "ockl.hpp"

extern "C" const __amdgpu_constant bool __oclc_finite_only_opt;
extern "C" const __amdgpu_constant bool __oclc_unsafe_math_opt;
extern "C" const __amdgpu_constant bool __oclc_daz_opt;
extern "C" const __amdgpu_constant bool __oclc_correctly_rounded_sqrt32;
extern "C" const __amdgpu_constant bool __oclc_wavefrontsize64;
extern "C" const __amdgpu_constant int __oclc_ISA_version;
extern "C" const __amdgpu_constant int __oclc_ABI_version;

#endif
