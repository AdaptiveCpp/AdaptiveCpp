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
#include "hipSYCL/sycl/libkernel/sscp/builtins/print.hpp"

extern "C" __acpp_uint64 __ockl_fprintf_stdout_begin();
extern "C" __acpp_uint64
__ockl_fprintf_append_string_n(__acpp_uint64 msg_desc, const char *data,
                               __acpp_uint64 length,
                               __acpp_uint32 is_last);

void __acpp_sscp_print(const char* msg) {
  constexpr int max_len = 1 << 16;

  int length = 0;
  while(length < max_len) {
    if(msg[length] == '\0') {
      // string length needs to include the null terminator
      ++length;
      break; 
    } else {
      ++length;
    }
  }

  auto handle = __ockl_fprintf_stdout_begin();
  __ockl_fprintf_append_string_n(handle, msg, static_cast<__acpp_uint64>(length), 1);
}
