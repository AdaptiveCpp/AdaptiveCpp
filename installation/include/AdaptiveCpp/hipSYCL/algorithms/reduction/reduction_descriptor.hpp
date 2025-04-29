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
#include <type_traits>

#ifndef HIPSYCL_REDUCTION_DESCRIPTOR_HPP
#define HIPSYCL_REDUCTION_DESCRIPTOR_HPP

namespace hipsycl::algorithms::reduction {


using initialization_flag_t = int;

namespace detail {

template <class ReductionDescriptor>
void set_reduction_result(
    const ReductionDescriptor &desc,
    typename ReductionDescriptor::value_type result,
    // may be unused if identity is known. For unknown identities, this
    // stores whether there have been contributions from the reduction.
    bool is_result_initialized) {
  using value_type = typename ReductionDescriptor::value_type;

  value_type *output = desc.get_final_output_destination();

  if (desc.has_init_value()) {
    if constexpr (ReductionDescriptor::has_known_identity()) {
      *output = desc.get_operator()(desc.get_init_vaue(), result);
    } else {
      if (is_result_initialized) {
        *output = desc.get_operator()(desc.get_init_vaue(), result);
      } else {
        *output = desc.get_init_vaue();
      }
    }
  } else { // no init value
    if constexpr (ReductionDescriptor::has_known_identity()) {
      *output = desc.get_operator()(*output, result);
    } else {
      if (is_result_initialized)
        *output = desc.get_operator()(*output, result);
    }
  }
}
}

/// Describes a user-provided operator for the reduction
template<class T, class BinaryOp, bool IsIdentityKnown=true>
struct reduction_binary_operator {
  reduction_binary_operator(BinaryOp op, const T& identity) noexcept
  : _op{op}, _identity{identity} {}

  using value_type = T;
  using binary_operation = BinaryOp;

  static constexpr bool has_known_identity() noexcept { return true; }

  T operator()(const T& a, const T& b) const noexcept { return _op(a, b); }

  T get_identity() const noexcept {
    return _identity;
  }

private:
  BinaryOp _op;
  T _identity;
};

template<class T, class BinaryOp>
struct reduction_binary_operator<T, BinaryOp, false> {
  reduction_binary_operator(BinaryOp op) noexcept
  : _op{op} {}

  using value_type = T;
  using binary_operation = BinaryOp;

  static constexpr bool has_known_identity() noexcept { return false; }

  T operator()(const T& a, const T& b) const noexcept { return _op(a, b); }

  T get_identity() const noexcept {
    return T{};
  }
private:
  BinaryOp _op;
};

/// In addition to the operator, provides information about
/// the reduction as configured by the user. This object
/// will be provided by the user.
template<class ReductionOperator, class OutputT>
class reduction_descriptor {
public:
  using value_type = typename ReductionOperator::value_type;
  using op_type = ReductionOperator;

  static constexpr bool has_known_identity() noexcept {
    return ReductionOperator::has_known_identity();
  }

  reduction_descriptor(const ReductionOperator &op,
                       const OutputT &output)
      : _op{op}, _output{output},
        _has_initialization_value{false} {}

  reduction_descriptor(const ReductionOperator &op,
                       const value_type &initialization_value,
                       const OutputT &output)
      : _op{op}, _output{output}, _initialization_value{initialization_value},
        _has_initialization_value{true} {}

  // Should only be called inside device code
  value_type* get_final_output_destination() const noexcept {
    if constexpr(std::is_pointer_v<OutputT>)
      return _output;
    else
      return _output.get_pointer();
  }

  bool has_init_value() const noexcept {
    return _has_initialization_value;
  }

  value_type get_init_vaue() const noexcept {
    return _initialization_value;
  }

  const ReductionOperator& get_operator() const noexcept {
    return _op;
  }
private:
  ReductionOperator _op;
  OutputT _output;
  value_type _initialization_value;
  bool _has_initialization_value;
};


}

#endif
