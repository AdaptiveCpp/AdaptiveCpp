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

#ifndef ACPP_GLUE_FCALL_SPECIALIZATION_HPP
#define ACPP_GLUE_FCALL_SPECIALIZATION_HPP

#include <string>
#include <vector>


namespace hipsycl::glue::sscp {

template <class T>
struct __acpp_sscp_emit_param_type_annotation_fcall_specialized_config {
  T value;
};

struct fcall_specialized_config {
  std::size_t unique_hash = 0;
  std::vector<std::pair<std::string, std::vector<std::string>>> function_call_map;
};

using fcall_config_kernel_property_t =
    __acpp_sscp_emit_param_type_annotation_fcall_specialized_config<
        const fcall_specialized_config *>;
}

#endif
