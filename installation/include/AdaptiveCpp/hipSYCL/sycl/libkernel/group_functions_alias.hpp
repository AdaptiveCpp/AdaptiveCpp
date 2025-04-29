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

#ifndef ACPP_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP
#define ACPP_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP

#include "backend.hpp"
#include "group_functions.hpp"

namespace hipsycl {
namespace sycl {

// any_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'any_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_any_of(Group g, T x, Predicate pred) {
  return any_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'any_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_any_of(Group g, bool pred) {
  return any_of_group(g, pred);
}

// all_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'all_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_all_of(Group g, T x, Predicate pred) {
  return all_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'all_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_all_of(Group g, bool pred) {
  return all_of_group(g, pred);
}

// none_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'none_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_none_of(Group g, T x, Predicate pred) {
  return none_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'none_of_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
bool group_none_of(Group g, bool pred) {
  return none_of_group(g, pred);
}

// reduce
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'reduce_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'reduce_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

// exclusive_scan
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'exclusive_scan_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'exclusive_scan_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'inclusive_scan_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_inclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, init, binary_op);
}
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'inclusive_scan_over_group' in SYCL 2020 Specification")]]
ACPP_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

} // namespace sycl
} // namespace hipsycl

#endif // ACPP_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP
