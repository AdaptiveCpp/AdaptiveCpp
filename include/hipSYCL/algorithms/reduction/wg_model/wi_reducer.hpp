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
  using combiner_type = typename ReductionBinaryOp::combiner_type;
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
