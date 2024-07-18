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
#ifndef HIPSYCL_SYCL_REDUCTION_HPP
#define HIPSYCL_SYCL_REDUCTION_HPP

#include <type_traits>
#include "backend.hpp"
#include "functional.hpp"
#include "accessor.hpp"
#include "hipSYCL/sycl/property.hpp"

#include "hipSYCL/algorithms/reduction/reduction_descriptor.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

//struct reduction_binary_operator<T, BinaryOp, false>

template<class T, class BinaryOp>
auto construct_reduction_op(BinaryOp op) {
  if constexpr(has_known_identity_v<BinaryOp, T>) {
    return algorithms::reduction::reduction_binary_operator<T, BinaryOp, true>{
        op, sycl::known_identity<BinaryOp, T>::value};
  } else {
    return algorithms::reduction::reduction_binary_operator<T, BinaryOp, false>{op};
  }
}

template<class T, class BinaryOp>
auto construct_reduction_op(BinaryOp op, const T& identity) {
  return algorithms::reduction::reduction_binary_operator<T, BinaryOp, true>{
        op, identity};
}

} // namespace detail


namespace property::reduction {

class initialize_to_identity : public detail::reduction_property
{};

}

/// Reducer implementation, builds on \c BackendReducerImpl concept.
/// \c BackendReducerImpl concept:
///   - defines value_type for reduction data type
///   - defines binary_operation for the binary combiner operation
///   - defines value_type identity() const
///   - defines void combine(const value_type&)
template <class BackendReducerImpl>
class reducer {
public:
  using value_type       = typename BackendReducerImpl::value_type;
  using binary_operation = typename BackendReducerImpl::binary_operation;
  static constexpr int dimensions = 0; // TODO: For span reductions, this should be 1


  ACPP_KERNEL_TARGET
  reducer(const reducer &) = delete;

  ACPP_KERNEL_TARGET
  reducer(reducer&&) = delete;

  ACPP_KERNEL_TARGET
  reducer& operator= (const reducer&) = delete;

  ACPP_KERNEL_TARGET
  reducer& operator= (reducer&&) = delete;

  ACPP_KERNEL_TARGET
  reducer(BackendReducerImpl &impl)
      : _impl{impl} {}

  ACPP_KERNEL_TARGET
  reducer& combine(const value_type &partial) {
    _impl.combine(partial);
    return *this;
  }

  // TODO
  // /* Only available if Dimensions > 0 */
  // __unspecified__ &operator[](size_t index) const;

  /* Only available if identity value is known */
  value_type identity() const {
    return _impl.identity();
  }

private:
  BackendReducerImpl& _impl;
};

#define HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(T)                                   \
  class Op = typename reducer<BackendReducerImpl>::binary_operation,              \
        std::enable_if_t <                                                     \
                std::is_same_v<                                                \
                    Op,                                                        \
                    T<typename reducer<BackendReducerImpl>::value_type>> ||    \
                std::is_same_v <                                               \
                    Op,                                                        \
                    T<void>>                                                   \
                > * = nullptr

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::plus)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator+=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  return r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::multiplies)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator*=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  return r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_and)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator&=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  return r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_or)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator|=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  return r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_xor)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator^=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  return r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::plus)>
ACPP_KERNEL_TARGET
reducer<BackendReducerImpl>& operator++(reducer<BackendReducerImpl> &r) {
  return r.combine(1);
}


class handler;

template <typename AccessorT, typename BinaryOperation>
auto
reduction(AccessorT vars, BinaryOperation combiner, const property_list& propList = {}) {
  auto reduction_op =
      detail::construct_reduction_op<typename AccessorT::value_type, BinaryOperation>(
          combiner);

  if(propList.has_property<property::reduction::initialize_to_identity>()) {
    return algorithms::reduction::reduction_descriptor{reduction_op, reduction_op.get_identity(), vars};
  } else {
    return algorithms::reduction::reduction_descriptor{reduction_op, vars};
  } 
}

template <typename AccessorT, typename BinaryOperation>
auto
reduction(AccessorT vars, const typename AccessorT::value_type &identity,
          BinaryOperation combiner, const property_list& propList = {}) {
  auto reduction_op =
      detail::construct_reduction_op<typename AccessorT::value_type, BinaryOperation>(
          combiner, identity);
  
  if(propList.has_property<property::reduction::initialize_to_identity>()) {
    return algorithms::reduction::reduction_descriptor{reduction_op, identity, vars};
  } else {
    return algorithms::reduction::reduction_descriptor{reduction_op, vars};
  }
}

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler &cgh, BinaryOperation combiner,
               const property_list &propList = {}) {
  sycl::accessor acc{vars, cgh};
  return reduction(acc, combiner, propList);
}

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler& cgh,
                const typename BufferT::value_type& identity,
                BinaryOperation combiner,
                const property_list& propList = {}) {
  sycl::accessor acc{vars, cgh};
  return reduction(acc, identity, combiner, propList);
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, BinaryOperation combiner,
              const property_list &propList = {}) {

  auto reduction_op =
      detail::construct_reduction_op<T, BinaryOperation>(combiner);

  if(propList.has_property<property::reduction::initialize_to_identity>()) {
    return algorithms::reduction::reduction_descriptor{reduction_op, reduction_op.get_identity(), var};
  } else {
    return algorithms::reduction::reduction_descriptor{reduction_op, var};
  }
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, const T &identity, BinaryOperation combiner,
              const property_list &propList = {}) {
  
  auto reduction_op =
      detail::construct_reduction_op<T, BinaryOperation>(combiner, identity);

  if(propList.has_property<property::reduction::initialize_to_identity>()) {
    return algorithms::reduction::reduction_descriptor{reduction_op, identity, var};
  } else {
    return algorithms::reduction::reduction_descriptor{reduction_op, var};
  }
}

/* Unsupported until we have span

template <typename T, typename Extent, typename BinaryOperation>
__unspecified__ reduction(span<T, Extent> vars, BinaryOperation combiner);

template <typename T, typename Extent, typename BinaryOperation>
__unspecified__ reduction(span<T, Extent> vars, const T& identity, BinaryOperation combiner);
*/


}
} // namespace hipsycl

#endif
