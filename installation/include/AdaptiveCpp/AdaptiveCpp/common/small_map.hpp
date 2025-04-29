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
#include <algorithm>

#include "small_vector.hpp"

#ifndef HIPSYCL_COMMON_SMALL_MAP_HPP
#define HIPSYCL_COMMON_SMALL_MAP_HPP

namespace hipsycl {
namespace common {

template<class Key, class Value>
class small_map {
public:

  using value_type = std::pair<Key,Value>;
  using key_type = Key;
  using mapped_type = Value;

  std::size_t size() const {
    return _v.size();
  }

  auto begin() noexcept { return _v.begin(); }
  auto begin() const noexcept { return _v.begin(); }
  auto cbegin() const noexcept { return _v.cbegin(); }

  auto end() noexcept { return _v.end(); }
  auto end() const noexcept { return _v.end(); }
  auto cend() const noexcept { return _v.cend(); }

  auto find(const Key& k) const noexcept {
    return std::find_if(
        _v.cbegin(), _v.cend(),
        [&](const value_type &v) { return v.first == k; });
  }

  auto find(const Key& k) noexcept {
    return std::find_if(
        _v.begin(), _v.end(),
        [&](const value_type &v) { return v.first == k; });
  }

  bool contains(const Key& k) const noexcept {
    return find(k) != _v.cend();
  }

  Value& operator[](const Key& k) {
    auto existing = find(k);
    
    if(existing != _v.end())
      return existing->second;
    
    _v.push_back(std::make_pair(k, Value{}));
    return _v.back().second;
  }

  void clear() {
    _v.clear();
  }

private:
  common::auto_small_vector<std::pair<Key,Value>> _v;
};

}
}

#endif
