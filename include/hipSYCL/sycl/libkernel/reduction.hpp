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
#include "accessor.hpp"
#include "functional.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename T, typename BinaryOperation, bool KnownIdentity>
struct reduction_descriptor_base;

template<typename T, typename BinaryOperation>
struct reduction_descriptor_base<T, BinaryOperation, false> {
    using value_type = T;
    using combiner_type = BinaryOperation;

    constexpr static bool has_identity = false;

    BinaryOperation combiner;

    explicit reduction_descriptor_base(BinaryOperation combiner)
        : combiner(combiner) {}
};

template<typename T, typename BinaryOperation>
struct reduction_descriptor_base<T, BinaryOperation, true> {
    using value_type = T;
    using combiner_type = BinaryOperation;

    constexpr static bool has_identity = true;

    BinaryOperation combiner;
    T identity;
    bool initialize_to_identity;

    explicit reduction_descriptor_base(BinaryOperation combiner, T identity, bool initialize_to_identity)
        : combiner(combiner), identity(identity), initialize_to_identity(initialize_to_identity) {}
};

template <class T, typename BinaryOperation, bool KnownIdentity>
struct pointer_reduction_descriptor
    : public reduction_descriptor_base<T, BinaryOperation, KnownIdentity>
{
  HIPSYCL_UNIVERSAL_TARGET
  pointer_reduction_descriptor(T *output_data, BinaryOperation op)
      : reduction_descriptor_base<T, BinaryOperation, KnownIdentity>{op}
      , data{output_data} {}

  HIPSYCL_UNIVERSAL_TARGET
  pointer_reduction_descriptor(T *output_data, T identity, BinaryOperation op, bool initialize_to_identity)
      : reduction_descriptor_base<T, BinaryOperation, KnownIdentity>{op, identity, initialize_to_identity}
      , data{output_data} {}

  T *data;

  HIPSYCL_UNIVERSAL_TARGET
  T *get_pointer() const {
    return data;
  }
};

template <class T, typename BinaryOperation>
pointer_reduction_descriptor(T *, BinaryOperation) -> pointer_reduction_descriptor<T, BinaryOperation, false>;

template <class T, typename BinaryOperation>
pointer_reduction_descriptor(T *, T, BinaryOperation, bool) -> pointer_reduction_descriptor<T, BinaryOperation, true>;


template <class AccessorT, typename BinaryOperation, bool KnownIdentity>
struct accessor_reduction_descriptor
    : public reduction_descriptor_base<typename AccessorT::value_type, BinaryOperation, KnownIdentity>
{
  HIPSYCL_UNIVERSAL_TARGET
  accessor_reduction_descriptor(AccessorT output_data, BinaryOperation op)
      : reduction_descriptor_base<typename AccessorT::value_type, BinaryOperation, KnownIdentity>{op}
      , acc{output_data} {}

  HIPSYCL_UNIVERSAL_TARGET
  accessor_reduction_descriptor(AccessorT output_data, typename AccessorT::value_type identity, BinaryOperation op,
          bool initialize_to_identity)
      : reduction_descriptor_base<typename AccessorT::value_type, BinaryOperation, KnownIdentity>{op, identity,
          initialize_to_identity}
      , acc{output_data} {}

  AccessorT acc;

  HIPSYCL_UNIVERSAL_TARGET
  typename AccessorT::value_type *get_pointer() const {
    return acc.get_pointer();
  }
};

template <class AccessorT, typename BinaryOperation>
accessor_reduction_descriptor(AccessorT, BinaryOperation)
    -> accessor_reduction_descriptor<AccessorT, BinaryOperation, false>;

template <class AccessorT, typename BinaryOperation>
accessor_reduction_descriptor(AccessorT, typename AccessorT::value_type, BinaryOperation, bool)
    -> accessor_reduction_descriptor<AccessorT, BinaryOperation, true>;

} // namespace detail

namespace property {
namespace reduction {

class initialize_to_identity: public detail::property{};

} // namespace reduction
} // namespace property

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


template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler &cgh, BinaryOperation combiner, property_list prop_list = {}) {
  bool initialize_to_identity = prop_list.has_property<property::reduction::initialize_to_identity>();
  auto acc_prop_list = initialize_to_identity ? property_list{no_init} : property_list{};
  if constexpr (has_known_identity_v<BinaryOperation, typename BufferT::value_type>) {
    return detail::accessor_reduction_descriptor{accessor{vars, cgh, write_only, acc_prop_list},
        known_identity_v<BinaryOperation, typename BufferT::value_type>, combiner, initialize_to_identity};
  } else {
    return detail::accessor_reduction_descriptor{accessor{vars, cgh, write_only, acc_prop_list}, combiner};
  }
}

template <typename BufferT, typename BinaryOperation>
auto reduction(BufferT vars, handler &cgh, const typename BufferT::value_type &identity,
          BinaryOperation combiner, property_list prop_list = {}) {
  static_assert(!has_known_identity_v<BinaryOperation, typename BufferT::value_type>);
  bool initialize_to_identity = prop_list.has_property<property::reduction::initialize_to_identity>();
  auto acc_prop_list = initialize_to_identity ? property_list{no_init} : property_list{};
  return detail::accessor_reduction_descriptor{accessor{vars, cgh, write_only, acc_prop_list}, identity, combiner,
      initialize_to_identity};
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, BinaryOperation combiner, property_list prop_list = {}) {
  if constexpr (has_known_identity_v<BinaryOperation, T>) {
    return detail::pointer_reduction_descriptor{var, known_identity_v<BinaryOperation, T>, combiner,
        prop_list.has_property<property::reduction::initialize_to_identity>()};
  } else {
    return detail::pointer_reduction_descriptor{var, combiner};
  }
}

template <typename T, typename BinaryOperation>
auto reduction(T *var, const T &identity, BinaryOperation combiner, property_list prop_list = {}) {
  static_assert(!has_known_identity_v<BinaryOperation, T>);
  return detail::pointer_reduction_descriptor{var, identity, combiner,
      prop_list.has_property<property::reduction::initialize_to_identity>()};
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
