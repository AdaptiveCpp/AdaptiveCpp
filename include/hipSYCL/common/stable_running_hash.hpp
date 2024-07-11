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
#ifndef HIPSYCL_STABLE_RUNNING_HASH_HPP
#define HIPSYCL_STABLE_RUNNING_HASH_HPP

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace hipsycl {
namespace common {

class stable_running_hash {
  static uint64_t constexpr prime = 1099511628211ULL;
  static uint64_t constexpr offset = 14695981039346656037ULL;

  uint64_t value;

public:
  stable_running_hash() : value{offset} {}

  void operator()(const void *data, std::size_t size) {
    for(int i = 0; i < size; i += sizeof(uint64_t)) {
      uint64_t current = 0;
      int read_size = std::min(sizeof(uint64_t), size - i);
      std::memcpy(&current, (char*)data + i, read_size);
      value ^= current;
      value *= prime;
    }
  }

  uint64_t get_current_hash() const { return value; }
};


}
}

#endif
