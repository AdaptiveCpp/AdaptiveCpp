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
#ifndef HIPSYCL_MEMCPY_HPP
#define HIPSYCL_MEMCPY_HPP

#include <vector>
#include "../operations.hpp"
#include "../util.hpp"

namespace hipsycl {
namespace rt {

class backend_manager;

class memcpy_model
{
public:
  memcpy_model(backend_manager* mgr){}

  cost_type estimate_runtime_cost(const memory_location &source,
                                  const memory_location &dest,
                                  range<3> num_elements) const;

  memory_location
  choose_source(const std::vector<memory_location> &candidate_sources,
                const memory_location &target, range<3> num_elements) const;

};


}
}

#endif