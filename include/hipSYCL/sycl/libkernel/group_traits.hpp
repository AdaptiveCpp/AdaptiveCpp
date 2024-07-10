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

#ifndef ACPP_LIBKERNEL_GROUP_TRAITS_HPP
#define ACPP_LIBKERNEL_GROUP_TRAITS_HPP

#include "group.hpp"
#include "sub_group.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

template<class T>
struct is_group : public std::false_type {};

template<int Dim>
struct is_group<group<Dim>> : public std::true_type {};

template<>
struct is_group<sub_group> : public std::true_type {};

template<class T>
inline constexpr bool is_group_v = is_group<T>::value;

} // namespace sycl
} // namespace hipsycl

#endif // ACPP_LIBKERNEL_GROUP_TRAITS_HPP
