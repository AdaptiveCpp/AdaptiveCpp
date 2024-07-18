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

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t i) noexcept
  {
    return _data[i];
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t i) const noexcept
  {
    return _data[i];
  }

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return N;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array& other) const noexcept
  {
    for(size_t i = 0; i < N; ++i)
      if(_data[i] != other._data[i])
        return false;
    return true;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  // Zero initialise to make device_array constexpr-constructible
  T _data[N] = {0};
};

template<class T>
struct device_array<T, 0>
{
  using iterator = T*;
  using const_iterator = const T*;

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return 0;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array&) const noexcept
  {
    return true;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t) noexcept
  {
    return *reinterpret_cast<T*>(0);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t) const noexcept
  {
    return *reinterpret_cast<T*>(0);
  }
};

// Use ACPP_LIBKERNEL_IS_DEVICE_PASS to make sure that this is also
// enabled for backends with unified host-device pass
#if ACPP_LIBKERNEL_IS_DEVICE_PASS

template<class T>
struct device_array<T, 1>
{
  using iterator = T*;
  using const_iterator = const T*;

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return 1;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array& other) const noexcept
  {
    return _x == other._x;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t) noexcept
  {
    return _x;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t) const noexcept
  {
    return _x;
  }

  // Zero initialise to make device_array constexpr-constructible
  T _x = 0;
};

template<class T>
struct device_array<T, 2>
{
  using iterator = T*;
  using const_iterator = const T*;

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return 2;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array& other) const noexcept
  {
    return _x == other._x && _y == other._y;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t idx) noexcept
  {
    if(idx == 0)
      return _x;
    else
      return _y;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t idx) const noexcept
  {
    if(idx == 0)
      return _x;
    else
      return _y;
  }

  // Zero initialise to make device_array constexpr-constructible
  T _x = 0;
  T _y = 0;
};


template<class T>
struct device_array<T, 3>
{
  using iterator = T*;
  using const_iterator = const T*;

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return 3;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array& other) const noexcept
  {
    return _x == other._x && _y == other._y && _z == other._z;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t idx) noexcept
  {
    if(idx == 0)
      return _x;
    else if(idx == 1)
      return _y;
    else
      return _z;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t idx) const noexcept
  {
    if(idx == 0)
      return _x;
    else if(idx == 1)
      return _y;
    else
      return _z;
  }
  
  // Zero initialise to make device_array constexpr-constructible
  T _x = 0;
  T _y = 0;
  T _z = 0;
};


template<class T>
struct device_array<T, 4>
{
  using iterator = T*;
  using const_iterator = const T*;

  ACPP_UNIVERSAL_TARGET
  constexpr size_t size() const noexcept
  {
    return 4;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator== (const device_array& other) const noexcept
  {
    return _x == other._x && _y == other._y && _z == other._z && _w == other._w;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr bool operator!= (const device_array& other) const noexcept
  {
    return !(*this == other);
  }

  ACPP_UNIVERSAL_TARGET
  constexpr T& operator[] (size_t idx) noexcept
  {
    if(idx == 0)
      return _x;
    else if(idx == 1)
      return _y;
    else if(idx == 2)
      return _z;
    else
      return _w;
  }

  ACPP_UNIVERSAL_TARGET
  constexpr const T& operator[] (size_t idx) const noexcept
  {
    if(idx == 0)
      return _x;
    else if(idx == 1)
      return _y;
    else if(idx == 2)
      return _z;
    else
      return _w;
  }

  // Zero initialise to make device_array constexpr-constructible
  T _x = 0;
  T _y = 0;
  T _z = 0;
  T _w = 0;
};

#endif // ACPP_LIBKERNEL_IS_DEVICE_PASS

}
}
}

#endif
