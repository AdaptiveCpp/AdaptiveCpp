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
