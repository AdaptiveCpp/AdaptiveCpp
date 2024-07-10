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
#include <cstddef>
#include <cstdint>

#ifndef HIPSYCL_THREADING_CONFIGURED_REDUCTION_DESCRIPTOR_HPP
#define HIPSYCL_THREADING_CONFIGURED_REDUCTION_DESCRIPTOR_HPP

#include "../reduction_descriptor.hpp"
#include "cache_line.hpp"

namespace hipsycl::algorithms::reduction::threading_model {

template <class ReductionDescriptor>
class configured_reduction_descriptor : public ReductionDescriptor {
public:
  // intermediate stage reduction
  configured_reduction_descriptor(
      // The basic reduction descriptor
      const ReductionDescriptor &basic_descriptor,
      // Array describing whether the input was initialized
      cache_line_aligned<initialization_flag_t> *is_initialized,
      cache_line_aligned<typename ReductionDescriptor::value_type> *scratch)
      : ReductionDescriptor{basic_descriptor},
        _is_initialized{is_initialized}, _scratch{scratch} {}

  cache_line_aligned<initialization_flag_t> *
  get_initialization_state() const noexcept {
    return _is_initialized;
  }

  cache_line_aligned<typename ReductionDescriptor::value_type> *
  get_scratch() const noexcept {
    return _scratch;
  }

private:
  cache_line_aligned<initialization_flag_t> *_is_initialized;
  cache_line_aligned<typename ReductionDescriptor::value_type> *_scratch;
};

}


#endif
