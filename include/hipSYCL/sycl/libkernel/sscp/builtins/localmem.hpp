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
#ifndef HIPSYCL_SSCP_BUILTINS_LOCAL_MEMORY
#define HIPSYCL_SSCP_BUILTINS_LOCAL_MEMORY

#include "builtin_config.hpp"

#define __acpp_sscp_local static __attribute__((address_space(3)))

HIPSYCL_SSCP_BUILTIN
__attribute__((address_space(3))) void* __acpp_sscp_get_dynamic_local_memory();

#endif