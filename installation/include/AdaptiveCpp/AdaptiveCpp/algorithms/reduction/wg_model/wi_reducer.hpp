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
#ifndef HIPSYCL_REDUCTION_WG_MODEL_WI_REDUCER_HPP
#define HIPSYCL_REDUCTION_WG_MODEL_WI_REDUCER_HPP

/// This file defines work-item reducers, i.e. classes
/// that are responsible for handling the reduction within one work item.

#include "../reduction_descriptor.hpp"

namespace hipsycl::algorithms::reduction::wg_model {

namespace detail {

template<class T, bool HasKnownIdentity=true>
struct sequential_reducer_storage {
  T current_value;
};

template<class T>
struct sequential_reducer_storage<T, false> {
  T current_value;
  bool is_initialized = false;
};

}

template<class ReductionBinaryOp>
class sequential_reducer {
public:
  using operator_type = ReductionBinaryOp;
  using binary_operation = typename ReductionBinaryOp::binary_operation;
  using value_type = typename ReductionBinaryOp::value_type;
  static constexpr bool is_identity_known =
      ReductionBinaryOp::has_known_identity();

  sequential_reducer() = default;
  sequential_reducer(const sequential_reducer& other) = default;
  sequential_reducer(const ReductionBinaryOp& op) noexcept
  : _op{op} {
    if constexpr(is_identity_known) {
      _data.current_value = op.get_identity();
    }
  }

  // Only available if identity is known
  auto identity() const {
    return _op.get_identity();
  }

  void combine(const value_type& val) noexcept {
    if constexpr(is_identity_known) {
      _data.current_value = _op(_data.current_value, val);
    } else {
      if(!_data.is_initialized) {
        _data.current_value = val;
        _data.is_initialized = true;
      } else {
        _data.current_value = _op(_data.current_value, val);
      }
    }
  }

  const value_type& value() const {
    return _data.current_value;
  }

  initialization_flag_t is_initialized() const {
    if constexpr(is_identity_known) {
      return true;
    } else {
      return _data.is_initialized;
    }
  }
private:
  ReductionBinaryOp _op;
  detail::sequential_reducer_storage<value_type,
                                     ReductionBinaryOp::has_known_identity()>
      _data;
};

}

#endif
