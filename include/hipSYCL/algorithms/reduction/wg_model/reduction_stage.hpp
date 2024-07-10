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

#ifndef HIPSYCL_WG_REDUCTION_STAGE_HPP
#define HIPSYCL_WG_REDUCTION_STAGE_HPP

#include "../reduction_descriptor.hpp"

namespace hipsycl::algorithms::reduction::wg_model {

struct reduction_stage_data {
  initialization_flag_t *is_input_initialized;
  initialization_flag_t *is_output_initialized;
  void *stage_input;
  void *stage_output;
};

template<class HorizontalReducer>
struct reduction_stage {
  std::size_t wg_size;
  std::size_t num_groups;
  std::size_t global_size;

  using data = reduction_stage_data;

  // this should be initialized to have one entry
  // per reduction operation.
  common::auto_small_vector<reduction_stage_data> data_plan;
  HorizontalReducer reducer;
  // The amount of local memory that will be allocated
  // for dedicated reduction kernel launches. reducers may
  // overwrite this during setup.
  std::size_t local_mem = 0;
};

}


#endif
