/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_DEVICE_ARRAY_HPP
#define HIPSYCL_DEVICE_ARRAY_HPP

#include <cstddef>
#include <type_traits>
#include "hipSYCL/sycl/libkernel/backend.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

template<class T, size_t N>
struct device_array
{
  using iterator = T*;
  using const_iterator = const T*;

  HIPSYCL_UNIVERSAL_TARGET
  T& operator[] (size_t i) noexcept
  {
    return _data[i];
  }

  HIPSYCL_UNIVERSAL_TARGET
  const T& operator[] (size_t i) const noexcept
  {
    return _data[i];
  }

  HIPSYCL_UNIVERSAL_TARGET
  size_t size() const noexcept
  {
    return N;
  }

  HIPSYCL_UNIVERSAL_TARGET
  iterator begin() noexcept
  {
    return &(_data[0]);
  }

  HIPSYCL_UNIVERSAL_TARGET
  const_iterator begin() const noexcept
  {
    return &(_data[0]);
  }

  HIPSYCL_UNIVERSAL_TARGET
  iterator end() noexcept
  {
    return begin() + N;
  }

  HIPSYCL_UNIVERSAL_TARGET
  const_iterator end() const noexcept
  {
    return begin() + N;
  }

  HIPSYCL_UNIVERSAL_TARGET
  bool operator== (const device_array& other) const noexcept
  {
    for(size_t i = 0; i < N; ++i)
      if(_data[i] != other._data[i])
        return false;
    return true;
  }

  HIPSYCL_UNIVERSAL_TARGET
  bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  T _data [N];
};

template<class T>
struct device_array<T, 0>
{
  using iterator = T*;
  using const_iterator = const T*;

  HIPSYCL_UNIVERSAL_TARGET
  size_t size() const noexcept
  {
    return 0;
  }

  HIPSYCL_UNIVERSAL_TARGET
  bool operator== (const device_array&) const noexcept
  {
    return true;
  }

  HIPSYCL_UNIVERSAL_TARGET
  bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  HIPSYCL_UNIVERSAL_TARGET
  T& operator[] (size_t) noexcept
  {
    return *reinterpret_cast<T*>(0);
  }

  HIPSYCL_UNIVERSAL_TARGET
  const T& operator[] (size_t) const noexcept
  {
    return *reinterpret_cast<T*>(0);
  }

  HIPSYCL_UNIVERSAL_TARGET
  iterator begin() noexcept
  {
    return nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  const_iterator begin() const noexcept
  {
    return nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  iterator end() noexcept
  {
    return nullptr;
  }

  HIPSYCL_UNIVERSAL_TARGET
  const_iterator end() const noexcept
  {
    return nullptr;
  }
};


}
}
}

#endif
