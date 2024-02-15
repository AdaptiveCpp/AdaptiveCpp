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

#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_get_subgroup_local_id() {
  return 0;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_get_subgroup_size() {
  return 1;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_get_subgroup_max_size() {
  return 1;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_get_subgroup_id() {
  size_t local_tid =
      __hipsycl_sscp_get_local_id_x() +
      __hipsycl_sscp_get_local_id_y() * (__hipsycl_sscp_get_local_size_x() +
      __hipsycl_sscp_get_local_id_z() * __hipsycl_sscp_get_local_size_x());
  return local_tid;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint32 __hipsycl_sscp_get_num_subgroups() {
  auto wg_size = __hipsycl_sscp_get_local_size_x() *
                 __hipsycl_sscp_get_local_size_y() *
                 __hipsycl_sscp_get_local_size_z();
  return wg_size;
}
