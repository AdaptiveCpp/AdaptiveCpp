/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_CODE_OBJECT_INVOKER_HPP
#define HIPSYCL_CODE_OBJECT_INVOKER_HPP

#include "error.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "util.hpp"
#include "kernel_cache.hpp"
#include "operations.hpp"

namespace hipsycl {
namespace rt {

class multipass_code_object_invoker {
public:
  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name_tag,
                               const std::string &kernel_body_name) = 0;
  virtual ~multipass_code_object_invoker(){}
};

class sscp_code_object_invoker {
public:
  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name,
                               const kernel_configuration& config) = 0;

  virtual rt::range<3> select_group_size(const rt::range<3> &global_range,
                                         const rt::range<3> &group_size) const {
    rt::range<3> selected_group_size = group_size;
    if(global_range[1] == 1 && global_range[2] == 1) {
      selected_group_size = rt::range<3>{128,1,1};
    } else if(global_range[2] == 1) {
      selected_group_size = rt::range<3>{16,16,1};
    } else {
      selected_group_size = rt::range<3>{8,8,4};
    }
    return selected_group_size;
  }
  
  virtual ~sscp_code_object_invoker(){}
};

}
} // namespace hipsycl

#endif
