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
#ifndef HIPSYCL_SPECIALIZED_HPP
#define HIPSYCL_SPECIALIZED_HPP

#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {

template <class T>
struct __acpp_sscp_emit_param_type_annotation_specialized {
  T value;
};

} // namespace detail

template <class T> class specialized {
public:
  template <typename U = T, typename = std::enable_if_t<
                                std::is_default_constructible<U>::value>>
  specialized() : _value{} {}

  specialized(const T &value) : _value{value} {}

  specialized(const specialized<T> &other) : _value{other._value.value} {}

  specialized(sycl::specialized<T> &&other) { swap(*this, other); }

  specialized<T> &operator=(const T &value) {
    sycl::specialized<T> tmp{value};
    swap(*this, tmp);
    return *this;
  }

  specialized<T> &operator=(specialized<T> other) {
    swap(*this, other);
    return *this;
  }

  friend void swap(specialized<T> &first, specialized<T> &second) {
    using std::swap;
    swap(first._value.value, second._value.value);
  }

  operator T() const { return _value.value; }

private:
  detail::__acpp_sscp_emit_param_type_annotation_specialized<T> _value;
};

} // namespace sycl
} // namespace hipsycl

#endif
