/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#include "CL/sycl/detail/scheduling/hw_model/memcpy.hpp"
#include <limits>


namespace cl {
namespace sycl {
namespace detail {


cost_type
memcpy_model::estimate_runtime_cost(const memory_location &source,
                                    const memory_location &dest,
                                    sycl::range<3> num_elements) const
{
  // Strongly prefer transfers from the same device to the same device
  if(source.get_device() == dest.get_device())
    return 1.0;

  if (source.get_device().get_full_backend_descriptor().hw_platform ==
      dest.get_device().get_full_backend_descriptor().hw_platform)
    return 2.0;

  return 3.0;
}

memory_location memcpy_model::choose_source(
    const std::vector<memory_location> &candidate_sources,
    const memory_location &target, sycl::range<3> num_elements) const 
{
  std::size_t best_transfer_index = 0;
  cost_type best_cost = std::numeric_limits<cost_type>::max();

  for (std::size_t i = 0; i < candidate_sources.size(); ++i)
  {
    cost_type cost = estimate_runtime_cost(candidate_sources[i], target, num_elements);
    if(cost < best_cost){
      best_cost = cost;
      best_transfer_index = i;
    }
  }
  return candidate_sources[best_transfer_index];
}
}
}
}
