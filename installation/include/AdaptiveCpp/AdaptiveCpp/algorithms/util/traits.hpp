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
#include <iterator>
#include <vector>
#include <array>
#include <type_traits>

#ifndef HIPSYCL_ALGORITHMS_UTIL_TRAITS_HPP
#define HIPSYCL_ALGORITHMS_UTIL_TRAITS_HPP

namespace hipsycl::algorithms::util {

template<class I>
constexpr bool is_contiguous() {
  // Cannot use contiguous iterator concept in C++17 :(
  using value_type = typename std::iterator_traits<I>::value_type;
  using vector_iterator =
      std::decay_t<decltype(std::vector<value_type>{}.begin())>;
  // TODO: Array, valarray, vector with custom allocator...
  return std::is_pointer_v<I> || (std::is_same_v<vector_iterator, I> &&
                                  !std::is_same_v<value_type, bool>);
}

}

#endif
