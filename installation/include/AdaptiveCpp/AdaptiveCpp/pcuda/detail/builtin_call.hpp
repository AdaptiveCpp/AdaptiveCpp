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

#ifndef ACPP_PCUDA_BUILTIN_CALL_HPP
#define ACPP_PCUDA_BUILTIN_CALL_HPP

#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"

#define PCUDA_BUILTIN_CALL(builtin) if(__acpp_sscp_is_device){builtin;}
#define PCUDA_BUILTIN_CALL_RESULT(builtin, fallback)                           \
  (__acpp_sscp_is_device ? (builtin) : (fallback))

#endif

