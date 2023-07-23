
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
  std::vector<ReductionStage> _stages;
  std::tuple<ReductionDescriptors...> _descriptors;
};

}


#endif
