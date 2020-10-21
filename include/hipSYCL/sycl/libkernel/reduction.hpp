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

namespace hipsycl {
namespace sycl {

namespace detail {

template <class T, typename BinaryOperation>
struct pointer_reduction_descriptor {
  T* data;
  T identity;
  BinaryOperation combiner;

  T *get_pointer() {
    return data;
  }
};

template <class AccessorT, typename BinaryOperation>
struct accessor_reduction_descriptor {
  AccessorT acc;
  typename AccessorT::value_type identity;
  BinaryOperation combiner;

  typename AccessorT::value_type *get_pointer() {
    return acc.get_pointer();
  }
};

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
