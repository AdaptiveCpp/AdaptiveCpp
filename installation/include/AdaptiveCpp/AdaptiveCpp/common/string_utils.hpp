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
#ifndef HIPSYCL_COMMON_STRING_UTILS_HPP
#define HIPSYCL_COMMON_STRING_UTILS_HPP

#include <vector>
#include <string>
#include <sstream>

namespace hipsycl {
namespace common {

inline std::vector<std::string> split_by_delimiter(const std::string &str, char delim,
                                            bool include_empty = true) {

  std::istringstream istream{str};

  std::vector<std::string> result;
  std::string current;
  while(std::getline(istream, current, delim)) {
    if(current.empty() && !include_empty)
      continue;
    result.push_back(current);
  }
  return result;
}

}
}

#endif
