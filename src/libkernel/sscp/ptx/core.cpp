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

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_x() {
  return __nvvm_read_ptx_sreg_tid_x();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_y() {
  return __nvvm_read_ptx_sreg_tid_y();;
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_id_z() {
  return __nvvm_read_ptx_sreg_tid_z();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_x() {
  return __nvvm_read_ptx_sreg_ctaid_x();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_y() {
  return __nvvm_read_ptx_sreg_ctaid_y();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_group_id_z() {
  return __nvvm_read_ptx_sreg_ctaid_z();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_x() {
  return __nvvm_read_ptx_sreg_ntid_x();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_y() {
  return __nvvm_read_ptx_sreg_ntid_y();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_local_size_z() {
  return __nvvm_read_ptx_sreg_ntid_z();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_x() {
  return __nvvm_read_ptx_sreg_nctaid_x();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_y() {
  return __nvvm_read_ptx_sreg_nctaid_y();
}

HIPSYCL_SSCP_BUILTIN __hipsycl_uint64 __hipsycl_sscp_get_num_groups_z() {
  return __nvvm_read_ptx_sreg_nctaid_z();
}
