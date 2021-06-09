/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and Contributors
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

#ifndef HIPSYCL_GLUE_REDUCTION_ACCUMULATOR_H
#define HIPSYCL_GLUE_REDUCTION_ACCUMULATOR_H

#include "hipSYCL/sycl/libkernel/backend.hpp"

namespace hipsycl {
namespace glue {

template<class ReductionDescriptor, class Enable = void>
struct sequential_reduction_accumulator;

template<class ReductionDescriptor>
struct sequential_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<ReductionDescriptor::has_identity>> {
  using value_type = typename ReductionDescriptor::value_type;

  value_type value;

  explicit HIPSYCL_UNIVERSAL_TARGET sequential_reduction_accumulator(
      const ReductionDescriptor &desc)
    : value(desc.identity) {}

  explicit HIPSYCL_UNIVERSAL_TARGET sequential_reduction_accumulator(value_type value)
    : value{value} {}

  HIPSYCL_UNIVERSAL_TARGET void combine_with(const ReductionDescriptor &desc, const value_type &v) {
    value = desc.combiner(value, v);
  }

  HIPSYCL_UNIVERSAL_TARGET void combine_with(const ReductionDescriptor &desc,
      const sequential_reduction_accumulator &v) {
    value = desc.combiner(value, v.value);
  }
};

template<class ReductionDescriptor>
struct sequential_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<!ReductionDescriptor::has_identity>> {
  using value_type = typename ReductionDescriptor::value_type;

  value_type value;
  bool initialized = false;

  explicit HIPSYCL_UNIVERSAL_TARGET sequential_reduction_accumulator(const ReductionDescriptor &) {}

  explicit HIPSYCL_UNIVERSAL_TARGET sequential_reduction_accumulator(value_type value,
      bool initialized)
    : value{value}, initialized{initialized} {}

  HIPSYCL_UNIVERSAL_TARGET void combine_with(const ReductionDescriptor &desc, const value_type &v) {
    value = initialized ? desc.combiner(value, v) : v;
    initialized = true;
  }

  HIPSYCL_UNIVERSAL_TARGET void combine_with(const ReductionDescriptor &desc,
      const sequential_reduction_accumulator &v) {
    if (!v.initialized) return;
    value = initialized ? desc.combiner(value, v.value) : v.value;
    initialized = true;
  }
};

}
}

#endif
