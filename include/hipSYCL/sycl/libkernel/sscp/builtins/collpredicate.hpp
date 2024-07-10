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

#ifndef HIPSYCL_SSCP_COLLECTIVE_PREDICATE_BUILTINS_HPP
#define HIPSYCL_SSCP_COLLECTIVE_PREDICATE_BUILTINS_HPP


HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool pred);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred);


HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred);

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred);

#endif
