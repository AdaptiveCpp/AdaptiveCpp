/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/half.hpp"
#include "hipSYCL/sycl/libkernel/detail/half_representation.hpp"

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__hipsycl_sscp_half_add(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_add(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__hipsycl_sscp_half_sub(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_sub(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__hipsycl_sscp_half_mul(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_mul(a,b);
}

HIPSYCL_SSCP_BUILTIN hipsycl::fp16::half_storage
__hipsycl_sscp_half_div(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_div(a,b);
}

HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_lt(hipsycl::fp16::half_storage a,
                       hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_less_than(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_lte(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_less_than_equal(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_gt(hipsycl::fp16::half_storage a,
                       hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_greater_than(a,b);
}
HIPSYCL_SSCP_BUILTIN bool
__hipsycl_sscp_half_gte(hipsycl::fp16::half_storage a,
                        hipsycl::fp16::half_storage b) {
  return hipsycl::fp16::builtin_greater_than(a,b);
}
