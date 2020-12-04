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

#ifndef HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP
#define HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP

#include "backend.hpp"
#include "group.hpp"
#include <type_traits>

#ifdef SYCL_DEVICE_ONLY
#include "generic/hiplike/group_functions.hpp"

#ifdef HIPSYCL_PLATFORM_CUDA
#include "cuda/group_functions.hpp"
#endif // HIPSYCL_PLATFORM_CUDA

#ifdef HIPSYCL_PLATFORM_HIP
#include "hip/group_functions.hpp"
#endif // HIPSYCL_PLATFORM_HIP

#endif // SYCL_DEVICE_ONLY

#include "host/group_functions.hpp"

namespace hipsycl {
namespace sycl {

// any_of
template<typename Group, typename T, typename Predicate,
         typename std::enable_if_t<!std::is_same_v<T, Predicate>, int> = 0>
HIPSYCL_KERNEL_TARGET
bool group_any_of(Group g, T x, Predicate pred) {
  return group_any_of(g, pred(x));
}


// all_of
template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool group_all_of(Group g, T x, Predicate pred) {
  return group_all_of(g, pred(x));
}


// none_of
template<typename Group, typename T, typename Predicate>
HIPSYCL_KERNEL_TARGET
bool group_none_of(Group g, T x, Predicate pred) {
  return group_none_of(g, pred(x));
}

// reduce
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_reduce(Group g, V x, T init, BinaryOperation binary_op) {
  T reduction = group_reduce(g, T{x}, binary_op);
  return binary_op(reduction, init);
}

// exclusive_scan
template<typename Group, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_exclusive_scan(Group g, T x, BinaryOperation binary_op) {
  return group_exclusive_scan(g, x, T{}, binary_op);
}

// inclusive_scan
template<typename Group, typename V, typename T, typename BinaryOperation>
HIPSYCL_KERNEL_TARGET
T group_inclusive_scan(Group g, V x, T init, BinaryOperation binary_op) {
  T scan = group_inclusive_scan(g, T{x}, binary_op);
  return binary_op(scan, init);
}


} // namespace sycl
} // namespace hipsycl

#endif // HIPSYCL_LIBKERNEL_GROUP_FUNCTIONS_HPP
