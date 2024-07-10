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
#ifndef HIPSYCL_ALGORITHMS_CACHE_LINE_HPP
#define HIPSYCL_ALGORITHMS_CACHE_LINE_HPP

#include <cstddef>

namespace hipsycl::algorithms::reduction::threading_model {

#ifndef HIPSYCL_FORCE_CACHE_LINE_SIZE
#define HIPSYCL_FORCE_CACHE_LINE_SIZE 128
#endif

#ifdef HIPSYCL_FORCE_CACHE_LINE_SIZE
constexpr std::size_t cache_line_size = HIPSYCL_FORCE_CACHE_LINE_SIZE;
#else
// This C++17 feature is unfortunately not yet widely supported
constexpr std::size_t cache_line_size =
    std::hardware_destructive_interference_size;
#endif

template <class T> struct cache_line_aligned {
  alignas(cache_line_size) T value;
};


}


#endif
