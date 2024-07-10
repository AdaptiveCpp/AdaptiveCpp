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
#include <tuple>

#ifndef HIPSYCL_ALGORITHMS_REDUCTION_PLAN_HPP
#define HIPSYCL_ALGORITHMS_REDUCTION_PLAN_HPP

namespace hipsycl::algorithms::reduction {

template<class ReductionStage, typename... ReductionDescriptors>
class reduction_plan {
public:
  reduction_plan(const ReductionDescriptors&... descriptors)
  : _descriptors{descriptors...} {}

  void push_back(const ReductionStage& stage) {
    _stages.push_back(stage);
  }

  auto begin() {
    return _stages.begin();
  }

  auto begin() const {
    return _stages.begin();
  }

  auto end() {
    return _stages.end();
  }

  auto end() const {
    return _stages.end();
  }


  ReductionStage& operator[](std::size_t i) {
    return _stages[i];
  }

  const ReductionStage& operator[](std::size_t i) const {
    return _stages[i];
  }

  std::size_t size() const {
    return _stages.size();
  }

  const std::tuple<ReductionDescriptors...> &get_descriptors() const {
    return _descriptors;
  }

private:
  common::auto_small_vector<ReductionStage> _stages;
  std::tuple<ReductionDescriptors...> _descriptors;
};

}


#endif
