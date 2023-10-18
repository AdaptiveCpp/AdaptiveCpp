/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay and contributors
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
