/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay
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

#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include <stddef.h>


size_t __spirv_BuiltInLocalInvocationId(int);
size_t __spirv_BuiltInWorkgroupId(int);
size_t __spirv_BuiltInWorkgroupSize(int);
size_t __spirv_BuiltInNumWorkgroups(int);

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_x() {
  return __spirv_BuiltInLocalInvocationId(0);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_y() {
  return __spirv_BuiltInLocalInvocationId(1);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_z() {
  return __spirv_BuiltInLocalInvocationId(2);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_x() {
  return __spirv_BuiltInWorkgroupId(0);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_y() {
  return __spirv_BuiltInWorkgroupId(1);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_z() {
  return __spirv_BuiltInWorkgroupId(2);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_x() {
  return __spirv_BuiltInWorkgroupSize(0);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_y() {
  return __spirv_BuiltInWorkgroupSize(1);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_z() {
  return __spirv_BuiltInWorkgroupSize(2);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_x() {
  return __spirv_BuiltInNumWorkgroups(0);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_y() {
  return __spirv_BuiltInNumWorkgroups(1);
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_z() {
  return __spirv_BuiltInNumWorkgroups(2);
}
