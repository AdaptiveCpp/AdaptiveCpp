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
#ifndef HIPSYCL_CODE_OBJECT_INVOKER_HPP
#define HIPSYCL_CODE_OBJECT_INVOKER_HPP

#include <string_view>

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
                               std::string_view kernel_name,
                               const rt::hcf_kernel_info* kernel_info,
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
