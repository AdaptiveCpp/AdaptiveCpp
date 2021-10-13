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

#ifndef HIPSYCL_VEC_HPP
#define HIPSYCL_VEC_HPP

#include "backend.hpp"
#include "multi_ptr.hpp"

#include <cstdint>
#include <type_traits>
#include <utility>

namespace hipsycl {
namespace sycl {
namespace detail {

template<class T, int N>
class vec_storage {
public:
  static constexpr std::size_t effective_size = (N == 3) ? 4 : N;
  static constexpr int alignment = effective_size * sizeof(T);

  using interop_type = vec_storage<T,N>;
  using value_type = T;

  HIPSYCL_UNIVERSAL_TARGET
  T& operator[](int i) { return _storage[i]; }

  HIPSYCL_UNIVERSAL_TARGET
  const T& operator[](int i) const { return _storage[i]; }

  template<int Index>
  HIPSYCL_UNIVERSAL_TARGET
  T& get() {
    return _storage[Index];
  }

  template<int Index>
  HIPSYCL_UNIVERSAL_TARGET
  const T& get() const {
    return _storage[Index];
  }

  template<class F>
  HIPSYCL_UNIVERSAL_TARGET
  void for_each(F&& f) {
    for(int i = 0; i < N; ++i)
      f(i, _storage[i]);
  }

  template<class F>
  HIPSYCL_UNIVERSAL_TARGET
  void for_each(F&& f) const {
    for(int i = 0; i < N; ++i)
      f(i, _storage[i]);
  }

  HIPSYCL_UNIVERSAL_TARGET
  interop_type interop() const {
    return *this;
  }

private:
  alignas(alignment) T _storage [effective_size];
};

// An alternative implementation of the vec_storage concept
// for swizzled access.
template<class TargetStorage, int... SwizzleIndices>
class swizzled_view_storage {
public:

  using interop_type = typename TargetStorage::interop_type;
  using value_type = typename TargetStorage::value_type;

  HIPSYCL_UNIVERSAL_TARGET
  swizzled_view_storage(TargetStorage& s)
  : _original_data{s} {}

  HIPSYCL_UNIVERSAL_TARGET
  value_type &operator[](int i) { 
    return _original_data[_swizzled_indices[i]];
  }

  HIPSYCL_UNIVERSAL_TARGET
  const value_type &operator[](int i) const {
    return _original_data[_swizzled_indices[i]];
  }

  template<int Index>
  HIPSYCL_UNIVERSAL_TARGET
  value_type& get() {
    return _original_data.template get<Index>();
  }

  template<int Index>
  HIPSYCL_UNIVERSAL_TARGET
  const value_type& get() const {
    return _original_data.template get<Index>();
  }

  template<class F>
  HIPSYCL_UNIVERSAL_TARGET
  void for_each(F&& f) {
    (f(SwizzleIndices, _original_data.template get<SwizzleIndices>()), ...);
  }

  template<class F>
  HIPSYCL_UNIVERSAL_TARGET
  void for_each(F&& f) const {
    (f(SwizzleIndices, _original_data.template get<SwizzleIndices>()), ...);
  }
  
  HIPSYCL_UNIVERSAL_TARGET
  interop_type interop() const {
    return _original_data.interop();
  }

private:
  static constexpr int _swizzled_indices[] = {SwizzleIndices...};
  TargetStorage& _original_data;
};

template<std::size_t N>
struct int_of_size {};

template <>
struct int_of_size<8> {
  using type = int8_t;
};
template <>
struct int_of_size<16> {
  using type = int16_t;
};
template <>
struct int_of_size<32> {
  using type = int32_t;
};
template <>
struct int_of_size<64> {
  using type = int64_t;
};

template<std::size_t N>
using int_of_size_t = typename int_of_size<N>::type;

template<class T>
struct logical_vector_op_result{
  using type = int_of_size_t<sizeof(T) * 8>;
};

template <class Vector_type, class Function>
HIPSYCL_UNIVERSAL_TARGET
void for_each_vector_element(Vector_type& v, Function&& f);

template <class Vector_type, class Function>
HIPSYCL_UNIVERSAL_TARGET
void for_each_vector_element(const Vector_type& v, Function&& f);

}

enum class rounding_mode {
  automatic, rte, rtz, rtp, rtn
};


struct elem
{
  static constexpr int x = 0;
  static constexpr int y = 1;
  static constexpr int z = 2;
  static constexpr int w = 3;
  static constexpr int r = 0;
  static constexpr int g = 1;
  static constexpr int b = 2;
  static constexpr int a = 3;
  static constexpr int s0 = 0;
  static constexpr int s1 = 1;
  static constexpr int s2 = 2;
  static constexpr int s3 = 3;
  static constexpr int s4 = 4;
  static constexpr int s5 = 5;
  static constexpr int s6 = 6;
  static constexpr int s7 = 7;
  static constexpr int s8 = 8;
  static constexpr int s9 = 9;
  static constexpr int sA = 10;
  static constexpr int sB = 11;
  static constexpr int sC = 12;
  static constexpr int sD = 13;
  static constexpr int sE = 14;
  static constexpr int sF = 15;
};


template <typename T, int N, class VectorStorage = detail::vec_storage<T, N>>
class vec {

  template <class Vector_type, class Function>
  HIPSYCL_UNIVERSAL_TARGET
  friend void detail::for_each_vector_element(Vector_type &v, Function &&f);

  template <class Vector_type, class Function>
  HIPSYCL_UNIVERSAL_TARGET
  friend void detail::for_each_vector_element(const Vector_type &v,
                                              Function &&f);

public:
  static_assert(N == 1 || N == 2 || N == 3 || N == 4 || N == 8 || N == 16,
                "Invalid number of vec elements");
  static_assert(std::is_same_v<bool, T> || std::is_same_v<char, T> ||
                    std::is_same_v<signed char, T> ||
                    std::is_same_v<unsigned char, T> ||
                    std::is_same_v<short int, T> ||
                    std::is_same_v<unsigned short int, T> ||
                    std::is_same_v<int, T> || std::is_same_v<unsigned int, T> ||
                    std::is_same_v<long int, T> ||
                    std::is_same_v<unsigned long int, T> ||
                    std::is_same_v<long long int, T> ||
                    std::is_same_v<unsigned long long int, T> ||
                    std::is_same_v<float, T> || std::is_same_v<double, T>,
                "Invalid data type for vec<>");

  using element_type = T;

  using vector_t = typename VectorStorage::interop_type;

  HIPSYCL_UNIVERSAL_TARGET
  vec(const VectorStorage& v)
  : _data{v} {}

  // The regular constructors are only available when we are not
  // a swizzled vector view
  template<class S = VectorStorage,
            std::enable_if_t<std::is_same_v<S, detail::vec_storage<T, N>>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  vec() {
    for(int i = 0; i < N; ++i)
      _data[i] = T{};
  }

  template <class S = VectorStorage,
            std::enable_if_t<std::is_same_v<S, detail::vec_storage<T, N>>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET explicit vec(const T &value) {
    for(int i = 0; i < N; ++i)
      _data[i] = value;
  }

  template <typename... Args, class S = VectorStorage,
            std::enable_if_t<std::is_same_v<S, detail::vec_storage<T, N>>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET vec(const Args &...args) {
    static_assert((count_num_elements<Args>() + ...) == N,
                  "Argument mismatch with vector size");

    int current_init_index = 0;
    (partial_initialization(current_init_index, args), ...);
  }

  template <class OtherStorage, class S = VectorStorage,
            std::enable_if_t<std::is_same_v<S, detail::vec_storage<T, N>>,
                             bool> = true>
  HIPSYCL_UNIVERSAL_TARGET vec(const vec<T, N, OtherStorage> &other)
  {
    for(int i = 0; i < N; ++i)
      _data[i] = other[i];
  }

  // Only available if vector_t is not already the storage
  // backend to avoid ambiguous overloads
  template <
      class Storage = VectorStorage,
      std::enable_if_t<!std::is_same_v<vector_t, Storage>, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET vec(vector_t v) : _data{v} {}

  HIPSYCL_UNIVERSAL_TARGET
  operator vector_t() const {
    return _data.interop();
  }

  template <int Dim = N, std::enable_if_t<Dim == 1, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  operator T() const { return _data.template get<0>(); }

  HIPSYCL_UNIVERSAL_TARGET
  static constexpr int get_count() { return N; }

  HIPSYCL_UNIVERSAL_TARGET
  static constexpr std::size_t get_size() { return sizeof(VectorStorage); }

  template <typename ConvertT, rounding_mode RM = rounding_mode::automatic>
  HIPSYCL_UNIVERSAL_TARGET
  vec<ConvertT, N> convert() const {

    vec<ConvertT, N> result;

    for(int i = 0; i < N; ++i) {
      // TODO: Take rounding mode into account
      result[i] = static_cast<ConvertT>(_data[i]);
    }

    return result;
  }

  template<typename AsT, int OtherN>
  HIPSYCL_UNIVERSAL_TARGET
  vec<AsT, OtherN> as() const {
    static_assert(sizeof(vec<AsT, OtherN>) == sizeof(vec<T,N>),
                  "Reinterpreted vector must have same size");
    static_assert(std::is_same_v<VectorStorage, detail::vec_storage<T, N>>,
                  "Reinterpreting swizzled vectors directly is not supported");

    vec<AsT, N> result;
    
    AsT* in_ptr = reinterpret_cast<AsT*>(&_data[0]);
    for(int i = 0; i < OtherN; ++i)
      result[i] = in_ptr[i];

    return result;
  }

  template<int... SwizzleIndices>
  HIPSYCL_UNIVERSAL_TARGET  
  auto swizzle() const {

    using swizzle_view_type =
        detail::swizzled_view_storage<VectorStorage, SwizzleIndices...>;

    VectorStorage& data = const_cast<VectorStorage&>(_data);

    return vec<T, sizeof...(SwizzleIndices), swizzle_view_type>{
        swizzle_view_type{data}};
  }

  template<int Dim = N, std::enable_if_t<(Dim > 1), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  auto lo() const {
    if constexpr(Dim == 1) {
      return swizzle<0>();
    } else if constexpr(Dim == 2) {
      return swizzle<0>();
    } else if constexpr(Dim == 3 || Dim == 4) {
      return swizzle<0,1>();
    } else if constexpr(Dim == 8) {
      return swizzle<0,1,2,3>();
    } else if constexpr(Dim == 16) {
      return swizzle<0,1,2,3,4,5,6,7>();
    } else {
      // Should never reach
      return *this;
    }
  }

  template<int Dim = N, std::enable_if_t<(Dim > 1), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  auto hi() const {
    if constexpr(Dim == 1) {
      return swizzle<0>();
    } else if constexpr(Dim == 2) {
      return swizzle<1>();
    } else if constexpr(Dim == 3 || Dim == 4) {
      return swizzle<2,3>();
    } else if constexpr(Dim == 8) {
      return swizzle<4,5,6,7>();
    } else if constexpr(Dim == 16) {
      return swizzle<8,9,10,11,12,13,14,15>();
    } else {
      // Should never reach
      return *this;
    }
  }

  template<int Dim = N, std::enable_if_t<(Dim > 1), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  auto odd() const {
    if constexpr(Dim == 1) {
      return swizzle<0>();
    } else if constexpr(Dim == 2) {
      return swizzle<1>();
    } else if constexpr(Dim == 3 || Dim == 4) {
      return swizzle<1,3>();
    } else if constexpr(Dim == 8) {
      return swizzle<1,3,5,7>();
    } else if constexpr(Dim == 16) {
      return swizzle<1,3,5,7,9,11,13,15>();
    } else {
      // Should never reach
      return *this;
    }
  }


  template<int Dim = N, std::enable_if_t<(Dim > 1), bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  auto even() const {
    if constexpr(Dim ==1) {
      return swizzle<0>();
    } else if constexpr(Dim == 2) {
      return swizzle<0>();
    } else if constexpr(Dim == 3 || Dim == 4) {
      return swizzle<0,2>();
    } else if constexpr(Dim == 8) {
      return swizzle<0,2,4,6>();
    } else if constexpr(Dim == 16) {
      return swizzle<0,2,4,6,8,10,12,14>();
    } else {
      // Should never reach
      return *this;
    }
  }


  HIPSYCL_UNIVERSAL_TARGET
  T& operator[](int index) {
    return _data[index];
  }

  HIPSYCL_UNIVERSAL_TARGET
  const T& operator[](int index) const {
    return _data[index];
  }

  template<class Storage>
  vec& operator=(const vec<T,N,Storage>& rhs) {
    for(int i = 0; i < N; ++i)
      _data[i] = rhs[i];
    return *this;
  }

  vec& operator=(const T& rhs) {
    for(int i = 0; i < N; ++i)
      _data[i] = rhs;
    return *this;
  }

#define HIPSYCL_DEFINE_VECTOR_ACCESS_IF(condition, name, id)                   \
  template <int Dim = N,                                                       \
            std::enable_if_t<(id < N) && (condition), bool> = true>            \
  HIPSYCL_UNIVERSAL_TARGET T &name() {                                         \
    return _data.template get<id>();                                           \
  }                                                                            \
                                                                               \
  template <int Dim = N,                                                       \
            std::enable_if_t<(id < N) && (condition), bool> = true>            \
  HIPSYCL_UNIVERSAL_TARGET T name() const {                                    \
    return _data.template get<id>();                                           \
  }

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 4, x, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 4, y, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 4, z, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 4, w, 3)

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim == 4, r, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim == 4, g, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim == 4, b, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim == 4, a, 3)

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s0, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s1, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s2, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s3, 3)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s4, 4)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s5, 5)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s6, 6)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s7, 7)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s8, 8)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, s9, 9)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sA, 10)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sB, 11)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sC, 12)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sD, 13)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sE, 14)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(Dim <= 16, sF, 15)

#define HIPSYCL_DEFINE_VECTOR_SWIZZLE2(name, Dim, i0, i1) \
  template<int n = N, \
           std::enable_if_t<(n >= Dim && n <= 4), bool> = true> \
  HIPSYCL_UNIVERSAL_TARGET \
  auto name() const \
  {return swizzle<i0,i1>(); }

#define HIPSYCL_DEFINE_VECTOR_SWIZZLE3(name, Dim, i0, i1, i2) \
  template<int n = N, \
           std::enable_if_t<(n >= Dim && n <= 4), bool> = true> \
  HIPSYCL_UNIVERSAL_TARGET \
  auto name() const \
  {return swizzle<i0,i1,i2>(); }

#define HIPSYCL_DEFINE_VECTOR_SWIZZLE4(name, i0, i1, i2, i3) \
  template<int n = N, \
           std::enable_if_t<(n == 4)>* = nullptr> \
  HIPSYCL_UNIVERSAL_TARGET \
  auto name() const \
  {return swizzle<i0,i1,i2,i3>(); }

#ifdef SYCL_SIMPLE_SWIZZLES
 #include "detail/vec_simple_swizzles.hpp"
#endif

  // ToDo: These functions have changed signatures with SYCL 2020
  // ToDo: could use native vector types for load / store
  template <access::address_space AddressSpace>
  HIPSYCL_UNIVERSAL_TARGET
  void load(size_t offset, multi_ptr<const T, AddressSpace> ptr) {
    for(int i = 0; i < N; ++i)
      _data[i] = ptr.get()[offset * N + i];
  }

  template <access::address_space AddressSpace>
  HIPSYCL_UNIVERSAL_TARGET
  void store(size_t offset, multi_ptr<T, AddressSpace> ptr) const {
    for(int i = 0; i < N; ++i)
      ptr.get()[offset * N + i] = _data[i];
  }

#define HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(op, t)                            \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend vec<t, N> operator op(const vec &lhs, const vec &rhs) {               \
    vec<t, N> result;                                                          \
    for (int i = 0; i < N; ++i) {                                              \
      result._data[i] = lhs._data[i] op rhs._data[i];                          \
    }                                                                          \
    return result;                                                             \
  }

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(%, t)

  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(+, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(-, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(*, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(/, T)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(&, t)
  
  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(|, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(^, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(>>, t)
  
  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_VEC(<<, t)

  #define HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(op, t)                       \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend vec<t, N> operator op(const vec &lhs, const t& rhs) {                 \
    vec<t, N> result;                                                          \
    for (int i = 0; i < N; ++i) {                                              \
      result._data[i] = lhs._data[i] op rhs;                                   \
    }                                                                          \
    return result;                                                             \
  }

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(%, t)

  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(+, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(-, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(*, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(/, T)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(&, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(|, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(^, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(>>, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_VEC_SCALAR(<<, t)

#define HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(op, t)                         \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend vec<t, N> operator op(const t &lhs, const vec& rhs) {                 \
    vec<t, N> result;                                                          \
    for (int i = 0; i < N; ++i) {                                              \
      result._data[i] = lhs op rhs._data[i];                                   \
    }                                                                          \
    return result;                                                             \
  }

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(%, t)

  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(+, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(-, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(*, T)
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(/, T)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(&, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(|, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(^, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(>>, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_BINARY_VEC_OP_SCALAR_VEC(<<, t)

  #define HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(op, t)                         \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend vec& operator op(vec& lhs, const vec<t,N>& rhs) {                     \
    for (int i = 0; i < N; ++i) {                                              \
      lhs._data[i] op rhs._data[i];                                            \
    }                                                                          \
    return lhs;                                                                \
  }

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(%=, t)

  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(+=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(-=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(*=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(/=, T)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(&=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(|=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(^=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(>>=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_VEC(<<=, t)

  #define HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(op, t)                      \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend vec& operator op(vec& lhs, const t& rhs) {                            \
    for (int i = 0; i < N; ++i) {                                              \
      lhs._data[i] op rhs;                                                     \
    }                                                                          \
    return lhs;                                                                \
  }

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(%=, t)

  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(+=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(-=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(*=, T)
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(/=, T)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(&=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(|=, t)
  
  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(^=, t)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(>>=, t)
  
  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_DEFINE_INPLACE_VEC_OP_VEC_SCALAR(<<=, t)

  HIPSYCL_UNIVERSAL_TARGET
  friend vec& operator++(vec& v) {
    for(int i = 0; i < N; ++i)
      ++(v._data[i]);
    return v;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec& operator--(vec& v) {
    for(int i = 0; i < N; ++i)
      --(v._data[i]);
    return v;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec<T,N> operator++(vec& v, int) {
    vec<T,N> old = v;

    for(int i = 0; i < N; ++i)
      ++(v._data[i]);

    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec<T,N> operator--(vec& v, int) {
    vec<T,N> old = v;

    for(int i = 0; i < N; ++i)
      --(v._data[i]);

    return old;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec<T,N> operator+(const vec& v) {
    return v;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec<T,N> operator-(const vec& v) {
    vec<T,N> result;
    for(int i = 0; i < N; ++i) {
      result._data[i] = -(v._data[i]);
    }
    return result;
  }

#define HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(op)                                     \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend auto operator op(const vec &lhs, const vec &rhs) {                    \
    using return_t = typename detail::logical_vector_op_result<T>::type;       \
    vec<return_t, N> result;                                                   \
    for (int i = 0; i < N; ++i) {                                              \
      result[i] = static_cast<return_t>(lhs[i] op rhs[i]);                     \
    }                                                                          \
    return result;                                                             \
  }


  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(&&)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(||)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(==)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(!=)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(>)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(<)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(>=)
  HIPSYCL_LOGICAL_VEC_OP_VEC_VEC(<=)

#define HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(op)                                  \
  HIPSYCL_UNIVERSAL_TARGET                                                     \
  friend auto operator op(const vec &lhs, const T &rhs) {                      \
    using return_t = typename detail::logical_vector_op_result<T>::type;       \
    vec<return_t, N> result;                                                   \
    for (int i = 0; i < N; ++i) {                                              \
      result[i] = static_cast<return_t>(lhs[i] op rhs);                        \
    }                                                                          \
    return result;                                                             \
  }

  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(&&)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(||)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(==)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(!=)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(>)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(<)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(>=)
  HIPSYCL_LOGICAL_VEC_OP_VEC_SCALAR(<=)

  template <typename t = T,
            std::enable_if_t<std::is_integral_v<t>, bool> = true>
  HIPSYCL_UNIVERSAL_TARGET
  friend vec<t, N> operator~(const vec &v) {
    vec<t, N> result;

    for (int i = 0; i < N; ++i) {
      result[i] = ~(v[i]);
    }

    return result;
  }

  HIPSYCL_UNIVERSAL_TARGET
  friend vec<typename detail::logical_vector_op_result<T>::type, N>
  operator!(const vec &v) {
    vec<typename detail::logical_vector_op_result<T>::type, N> result;
    for (int i = 0; i < N; ++i)
      result[i] = !static_cast<bool>(v[i]);
    return result;
  }
private:
  template <typename Arg>
  HIPSYCL_UNIVERSAL_TARGET
  static constexpr int count_num_elements() {
    if constexpr(std::is_scalar_v<Arg>)
      return 1;
    else if(std::is_same_v<typename Arg::element_type, T>)
      return Arg::get_count();
    // ToDo: Trigger error
    return 0;
  }

  template<typename Arg>
  HIPSYCL_UNIVERSAL_TARGET
  void partial_initialization(int& current_init_index, const Arg& x) {
    if constexpr(std::is_scalar_v<Arg>) {
      _data[current_init_index] = x;
      ++current_init_index;
    } else {
      // Assume we are dealing with another vector
      constexpr int count = count_num_elements<Arg>();
      
      for(int i = 0; i < count; ++i) {
        _data[i + current_init_index] = x[i];
      }
      
      current_init_index += count;
    }
  }

  VectorStorage _data;
};

using char2 = vec<int8_t, 2>;
using char3 = vec<int8_t, 3>;
using char4 = vec<int8_t, 4>;
using char8 = vec<int8_t, 8>;
using char16 = vec<int8_t, 16>;

using uchar2 = vec<uint8_t, 2>;
using uchar3 = vec<uint8_t, 3>;
using uchar4 = vec<uint8_t, 4>;
using uchar8 = vec<uint8_t, 8>;
using uchar16 = vec<uint8_t, 16>;

using short2 = vec<int16_t, 2>;
using short3 = vec<int16_t, 3>;
using short4 = vec<int16_t, 4>;
using short8 = vec<int16_t, 8>;
using short16 = vec<int16_t, 16>;

using ushort2 = vec<uint16_t, 2>;
using ushort3 = vec<uint16_t, 3>;
using ushort4 = vec<uint16_t, 4>;
using ushort8 = vec<uint16_t, 8>;
using ushort16 = vec<uint16_t, 16>;

using int2 = vec<int32_t, 2>;
using int3 = vec<int32_t, 3>;
using int4 = vec<int32_t, 4>;
using int8 = vec<int32_t, 8>;
using int16 = vec<int32_t, 16>;

using uint2 = vec<uint32_t, 2>;
using uint3 = vec<uint32_t, 3>;
using uint4 = vec<uint32_t, 4>;
using uint8 = vec<uint32_t, 8>;
using uint16 = vec<uint32_t, 16>;

using long2 = vec<int64_t, 2>;
using long3 = vec<int64_t, 3>;
using long4 = vec<int64_t, 4>;
using long8 = vec<int64_t, 8>;
using long16 = vec<int64_t, 16>;

using ulong2 = vec<uint64_t, 2>;
using ulong3 = vec<uint64_t, 3>;
using ulong4 = vec<uint64_t, 4>;
using ulong8 = vec<uint64_t, 8>;
using ulong16 = vec<uint64_t, 16>;

using float2 = vec<float, 2>;
using float3 = vec<float, 3>;
using float4 = vec<float, 4>;
using float8 = vec<float, 8>;
using float16 = vec<float, 16>;

using double2 = vec<double, 2>;
using double3 = vec<double, 3>;
using double4 = vec<double, 4>;
using double8 = vec<double, 8>;
using double16 = vec<double, 16>;

namespace detail {


template <class Vector_type, class Function>
HIPSYCL_UNIVERSAL_TARGET
void for_each_vector_element(Vector_type &v, Function &&f) {
  v._data.for_each(f);
}

template <class Vector_type, class Function>
HIPSYCL_UNIVERSAL_TARGET
void for_each_vector_element(const Vector_type &v, Function &&f) {
  v._data.for_each(f);
}

}

}
}

#endif
