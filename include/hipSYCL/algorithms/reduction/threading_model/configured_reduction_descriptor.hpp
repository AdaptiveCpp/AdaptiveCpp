
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
