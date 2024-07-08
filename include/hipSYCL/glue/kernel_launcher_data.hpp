/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_DATA_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_DATA_HPP

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/kernel_type.hpp"
#include "hipSYCL/runtime/error.hpp"

#include <vector>
#include <string_view>
#include <functional>

namespace hipsycl {

namespace sycl {
class interop_handle;
}

namespace rt {
class dag_node;
class kernel_configuration;
class backend_kernel_launch_capabilities;
class hcf_kernel_info;
}

namespace glue {

// some kernel launchers use this structure to store
// their data. This avoids having to malloc kernel launcher objects.
//
// This is currently only used by the SSCP kernel launcher. In the future
// we may want to investigate unifying the other kernel launchers into this
// data structure as well, which would allow to share data between them.
struct kernel_launcher_data {
  // will be configured during launch
  void* params = nullptr;

  // data potentially shared by multiple launchers
  rt::kernel_type type;
  // has to be mutable e.g. to initialize accessors (embedded pointers)
  mutable std::vector<uint8_t> kernel_args;
  rt::range<3> global_size; // <- indices must be flipped
  rt::range<3> group_size; // <- indices must be flipped
  unsigned local_mem_size;
  // In case the launch is a custom operation
  std::function<void(sycl::interop_handle&)> custom_op;

  using invoker_function_t = rt::result (*)(
      const kernel_launcher_data &launch_config, rt::dag_node *node,
      const rt::kernel_configuration &kernel_config,
      const rt::backend_kernel_launch_capabilities &launch_capabilities,
      void *backend_params);

  // compilation flow-specific fields
  unsigned long long sscp_hcf_object_id;
  const char* sscp_kernel_id = nullptr;
  const rt::hcf_kernel_info* kernel_info = nullptr;
  invoker_function_t sscp_invoker;
};



}
}

#endif
