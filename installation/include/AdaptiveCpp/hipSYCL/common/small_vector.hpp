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
#ifndef HIPSYCL_COMMON_SMALL_VECTOR_HPP
#define HIPSYCL_COMMON_SMALL_VECTOR_HPP

#include <memory>
#include "sbo/small_vector.hpp"

namespace hipsycl {
namespace common {

template<class T, int N, class Allocator = std::allocator<T>>
using small_vector = sbo::small_vector<T, N>;

template <class T, class Allocator = std::allocator<T>>
using auto_small_vector =
    sbo::small_vector<T, ((64 + sizeof(T) - 1)/ sizeof(T))>;
}
}

#endif
