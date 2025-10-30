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

// This container only has static storage, but it still
// tracks how many elements of the static storage are used up.
// If the available capacity is insufficient, then try_push_back()
// will fail.
// This is useful when cutting off data is acceptable, e.g. for an optimization
// that might just fail when the capacity is too small.
//
// This class is very minimalistic and does (by design) not implement
// the full vector API and semantics!
template<class T, int N>
class small_static_vector {
public:

  T* begin() {
    return &_data[0];
  }

  T* end() {
    return _data+_size;
  }

  const T* begin() const {
    return &_data[0];
  }

  const T* end() const {
    return _data+_size;
  }

  const T* cbegin() const noexcept {
    return &_data[0];
  }

  const T* cend() const noexcept {
    return _data+_size;
  }

  bool try_push_back(const T& x) {
    if(_size < N) {
      _data[_size] = x;
      ++_size;
      return true;
    }
    _is_insufficient = true;
    return false;
  }

  bool is_capacity_insufficient() const {
    return _is_insufficient;
  }

  static constexpr int capacity() {
    return N;
  }

  int size() const {
    return _size;
  }

  T& operator[](int i) {
    return _data[i];
  }

  const T& operator[](int i) const {
    return _data[i];
  }

  friend bool operator==(const small_static_vector &a,
                         const small_static_vector &b) {
    if(a._size != b._size)
      return false;
    for(int i = 0; i < a._size; ++i) {
      if(a._data[i] != b._data[i])
        return false;
    }
    return true;
  }

  friend bool operator!=(const small_static_vector &a,
                         const small_static_vector &b) {
    return !(a == b);
  }
private:
  T _data [N] = {};
  int _size = 0;
  bool _is_insufficient = false;
};

}
}

#endif
