/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_SYCL_REDUCTION_HPP
#define HIPSYCL_SYCL_REDUCTION_HPP

#include <type_traits>
#include "backend.hpp"
#include "functional.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template <class T, typename BinaryOperation>
struct pointer_reduction_descriptor {
  using value_type = T;
  using combiner_type = BinaryOperation;

  HIPSYCL_UNIVERSAL_TARGET
  pointer_reduction_descriptor(value_type *output_data, value_type op_identity,
                               BinaryOperation op)
      : data{output_data}, identity{op_identity}, combiner{op} {}

  value_type* data;
  value_type identity;
  BinaryOperation combiner;

  HIPSYCL_UNIVERSAL_TARGET
  value_type *get_pointer() const {
    return data;
  }
};

template <class AccessorT, typename BinaryOperation>
struct accessor_reduction_descriptor {
  using value_type = typename AccessorT::value_type;
  using combiner_type = BinaryOperation;

  HIPSYCL_UNIVERSAL_TARGET
  accessor_reduction_descriptor(AccessorT output_data, value_type op_identity,
                                BinaryOperation op)
      : acc{output_data}, identity{op_identity}, combiner{op} {}

  AccessorT acc;
  value_type identity;
  BinaryOperation combiner;

  HIPSYCL_UNIVERSAL_TARGET
  value_type *get_pointer() const {
    return acc.get_pointer();
  }
};

} // namespace detail

/// Reducer implementation, builds on \c BackendReducerImpl concept.
/// \c BackendReducerImpl concept:
///   - defines value_type for reduction data type
///   - defines combiner_type for the binary combiner operation
///   - defines value_type identity() const
///   - defines void combine(const value_type&)
template <class BackendReducerImpl> class reducer {
public:
  
  using value_type    = typename BackendReducerImpl::value_type;
  using combiner_type = typename BackendReducerImpl::combiner_type;

  HIPSYCL_KERNEL_TARGET
  reducer(const reducer &) = delete;

  HIPSYCL_KERNEL_TARGET
  reducer(reducer&&) = default;

  HIPSYCL_KERNEL_TARGET
  reducer(BackendReducerImpl &impl)
      : _impl{impl} {}

  HIPSYCL_KERNEL_TARGET
  reducer& operator= (const reducer&) = delete;

  HIPSYCL_KERNEL_TARGET
  void combine(const value_type &partial) {
    return _impl.combine(partial);
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
  class Op = typename reducer<BackendReducerImpl>::combiner_type,              \
  std::enable_if_t<std::is_same_v<                                             \
            Op, T<typename reducer<BackendReducerImpl>::value_type>>> * =      \
            nullptr

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::plus)>
HIPSYCL_KERNEL_TARGET
void operator+=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::multiplies)>
HIPSYCL_KERNEL_TARGET
void operator*=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_and)>
HIPSYCL_KERNEL_TARGET
void operator&=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_or)>
HIPSYCL_KERNEL_TARGET
void operator|=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::bit_xor)>
HIPSYCL_KERNEL_TARGET
void operator^=(reducer<BackendReducerImpl> &r,
                const typename reducer<BackendReducerImpl>::value_type &v) {
  r.combine(v);
}

template <class BackendReducerImpl,
          HIPSYCL_ENABLE_REDUCER_OP_IF_TYPE(sycl::plus)>
HIPSYCL_KERNEL_TARGET
void operator++(reducer<BackendReducerImpl> &r) {
  r.combine(1);
}


template <typename AccessorT, typename BinaryOperation>
detail::accessor_reduction_descriptor<AccessorT, BinaryOperation>
reduction(AccessorT vars, BinaryOperation combiner) {

  auto identity = typename AccessorT::value_type{};
  return detail::accessor_reduction_descriptor{vars, identity, combiner};
}

template <typename AccessorT, typename BinaryOperation>
detail::accessor_reduction_descriptor<AccessorT, BinaryOperation>
reduction(AccessorT vars, const typename AccessorT::value_type &identity,
          BinaryOperation combiner) {

  return detail::accessor_reduction_descriptor{vars, identity, combiner};
}

template <typename T, typename BinaryOperation>
detail::pointer_reduction_descriptor<T, BinaryOperation>
reduction(T *var, BinaryOperation combiner) {

  return detail::pointer_reduction_descriptor{var, T{}, combiner};
}

template <typename T, typename BinaryOperation>
detail::pointer_reduction_descriptor<T, BinaryOperation>
reduction(T *var, const T &identity, BinaryOperation combiner) {
  return detail::pointer_reduction_descriptor{var, identity, combiner};
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
