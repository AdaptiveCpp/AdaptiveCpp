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
#include "hipSYCL/runtime/hw_model/memcpy.hpp"
#include <limits>


namespace hipsycl {
namespace rt {


cost_type
memcpy_model::estimate_runtime_cost(const memory_location &source,
                                    const memory_location &dest,
                                    range<3> num_elements) const
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
    const memory_location &target, range<3> num_elements) const 
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
