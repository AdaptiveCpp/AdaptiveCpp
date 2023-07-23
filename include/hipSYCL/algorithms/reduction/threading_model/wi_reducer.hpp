/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_REDUCTION_THREADING_WI_REDUCER_HPP
#define HIPSYCL_REDUCTION_THREADING_WI_REDUCER_HPP

/// This file defines work-item reducers, i.e. classes
/// that are responsible for handling the reduction within one work item.

#include "../reduction_descriptor.hpp"
#include "cache_line.hpp"

namespace hipsycl::algorithms::reduction::threading_model {

/// This is mainly intended for the host parallel_for(range) case
template <class ReductionBinaryOp> class sequential_reducer {
public:
  using operator_type = ReductionBinaryOp;
  using value_type = typename ReductionBinaryOp::value_type;
  static constexpr bool is_identity_known =
      ReductionBinaryOp::has_known_identity();

  sequential_reducer() = default;
  sequential_reducer(
      const ReductionBinaryOp &op,
      cache_line_aligned<value_type> *current_value_location,
      cache_line_aligned<initialization_flag_t> *initialization_state) noexcept
      : _op{op}, _current_value{current_value_location},
        _initialization_state{initialization_state} {}

  void combine(const value_type &val) noexcept {
    if constexpr (is_identity_known) {
      _current_value->value = _op(_current_value->value, val);
    } else {
      if (!_initialization_state->value) {
        _initialization_state->value = true;
        _current_value->value = val;
      } else {
        _current_value->value = _op(_current_value->value, val);
      }
    }
  }

  const value_type &value() const { return *_current_value; }

  initialization_flag_t is_initialized() const {
    return is_identity_known || _initialization_state->value;
  }

private:
  ReductionBinaryOp _op;
  cache_line_aligned<value_type> *_current_value;
  cache_line_aligned<initialization_flag_t> *_initialization_state;
};

} // namespace hipsycl::algorithms::reduction::threading_model

#endif
