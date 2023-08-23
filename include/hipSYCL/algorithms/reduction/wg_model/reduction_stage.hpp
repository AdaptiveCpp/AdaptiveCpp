
/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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
  std::vector<reduction_stage_data> data_plan;
  HorizontalReducer reducer;
  // The amount of local memory that will be allocated
  // for dedicated reduction kernel launches. reducers may
  // overwrite this during setup.
  std::size_t local_mem = 0;
};

}


#endif
