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
#include "builtin_config.hpp"

#ifndef HIPSYCL_SSCP_SUBGROUP_BUILTINS_HPP
#define HIPSYCL_SSCP_SUBGROUP_BUILTINS_HPP

HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_local_id();
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_size();
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_max_size();
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_subgroup_id();
HIPSYCL_SSCP_BUILTIN __acpp_uint32 __acpp_sscp_get_num_subgroups();

#endif
