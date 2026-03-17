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

#include "hipSYCL/sycl/libkernel/sscp/builtins/collpredicate.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/reduction.hpp"

__acpp_int32 sub_group_all(__acpp_int32) __attribute__((convergent));
__acpp_int32 sub_group_any(__acpp_int32) __attribute__((convergent));

/*
 * OpenCL-C work-group collectives are not supported in clspv as they are an
 * OpenCL 3.0 feature, so instead use the internal SSCP wg reduce
 * implementation.
 */
HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_any(bool pred) {
  return __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op::logical_or,
                                          pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_all(bool pred) {
  return __acpp_sscp_work_group_reduce_i8(__acpp_sscp_algorithm_op::logical_and,
                                          pred);
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_work_group_none(bool pred) {
  bool result_or = __acpp_sscp_work_group_reduce_i8(
      __acpp_sscp_algorithm_op::logical_or, pred);
  return !result_or;
}

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_all(bool pred) { return sub_group_all(pred) != 0; }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_any(bool pred) { return sub_group_any(pred) != 0; }

HIPSYCL_SSCP_CONVERGENT_BUILTIN
bool __acpp_sscp_sub_group_none(bool pred) { return sub_group_any(pred) == 0; }
