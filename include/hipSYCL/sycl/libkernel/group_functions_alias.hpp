/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP
#define HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP

#include "backend.hpp"
#include "group_functions.hpp"

namespace hipsycl {
namespace sycl {

// any_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'any_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_any_of(Group g, T x, Predicate pred) {
  return any_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'any_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_any_of(Group g, bool pred) {
  return any_of_group(g, pred);
}

// all_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'all_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_all_of(Group g, T x, Predicate pred) {
  return all_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'all_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_all_of(Group g, bool pred) {
  return all_of_group(g, pred);
}

// none_of
template<typename Group, typename T, typename Predicate>
[[deprecated("renamed to 'none_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_none_of(Group g, T x, Predicate pred) {
  return none_of_group(g, x, pred);
}
template<typename Group>
[[deprecated("renamed to 'none_of_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
bool group_none_of(Group g, bool pred) {
  return none_of_group(g, pred);
}

// reduce
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'reduce_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, V x, T init, BinaryOperation binary_op) {
  return reduce_over_group(g, x, init, binary_op);
}
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'reduce_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, T x, BinaryOperation binary_op) {
  return reduce_over_group(g, x, binary_op);
}

// exclusive_scan
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'exclusive_scan_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, binary_op);
}
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'exclusive_scan_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return exclusive_scan_over_group(g, x, init, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
[[deprecated("renamed to 'inclusive_scan_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, init, binary_op);
}
template<typename Group, typename T, typename BinaryOperation>
[[deprecated("renamed to 'inclusive_scan_over_group' in SYCL 2020 Specification")]]
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return inclusive_scan_over_group(g, x, binary_op);
}

} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_ALIAS_HPP
