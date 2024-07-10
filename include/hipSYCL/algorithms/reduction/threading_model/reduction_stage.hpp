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
#include <vector>

#ifndef HIPSYCL_THREADING_REDUCTION_STAGE_HPP
#define HIPSYCL_THREADING_REDUCTION_STAGE_HPP

#include "../reduction_descriptor.hpp"
#include "cache_line.hpp"

namespace hipsycl::algorithms::reduction::threading_model {


struct reduction_stage_data {
  cache_line_aligned<initialization_flag_t> *is_data_initialized;
  void *scratch_data;
};

struct reduction_stage {
  std::size_t global_size;

  // this should be initialized to have one entry
  // per reduction operation.
  common::auto_small_vector<reduction_stage_data> data_plan;
};

}


#endif
