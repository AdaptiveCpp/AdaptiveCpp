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

#include <cstddef>
#include <type_traits>
#include "backend/backend.hpp"
#include "types.hpp"
#include "access.hpp"
#include "multi_ptr.hpp"
#include "detail/vec.hpp"

namespace cl {
namespace sycl {
namespace detail {


template<class T, int N, class Function>
__host__ __device__
inline void transform_vector(vec<T,N>& v, Function f);

template<class T, int N, class Function>
__host__ __device__
inline vec<T,N> binary_vector_operation(const vec<T,N>& a,
                                        const vec<T,N>& b,
                                        Function f);

template<class T, int N, class Function>
__host__ __device__
inline vec<T,N> trinary_vector_operation(const vec<T,N>& a,
                                         const vec<T,N>& b,
                                         const vec<T,N>& c,
                                         Function f);




} // detail


enum class rounding_mode
{
  automatic,
  rte,
  rtz,
  rtp,
  rtn
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

template <typename dataT, int numElements>
class vec
{
  template<class T, int N, class Function>
  __host__ __device__
  friend void detail::transform_vector(vec<T,N>& v,
                                       Function f);

  template<class T, int N, class Function>
  __host__ __device__
  friend vec<T,N> detail::binary_vector_operation(const vec<T,N>& a,
                                                  const vec<T,N>& b,
                                                  Function f);

  template<class T, int N, class Function>
  __host__ __device__
  friend vec<T,N> detail::trinary_vector_operation(const vec<T,N>& a,
                                                   const vec<T,N>& b,
                                                   const vec<T,N>& c,
                                                   Function f);

  template<class T, int N>
  friend class vec;

  __host__ __device__
  explicit vec(const detail::vector_impl<dataT,numElements>& v)
    : _impl{v}
  {}

  /// Initialization from numElements scalars
  template<int... Indices, typename... Args>
  __host__ __device__
  void scalar_init(detail::vector_index_sequence<Indices...>, Args... args)
  {
    static_assert(sizeof...(args) == numElements,
                  "Invalid number of arguments for vector "
                  "initialization");

    auto dummy_initializer = { ((_impl.template set<Indices>(args)), 0)... };
  }

  /// Initialization from single scalar
  template<int... Indices>
  __host__ __device__
  void broadcast_scalar_init(detail::vector_index_sequence<Indices...>, dataT x)
  {
    auto dummy_initializer = { ((_impl.template set<Indices>(x)), 0)... };
  }

  template<int Start_index, int N, int... Indices>
  __host__ __device__
  constexpr void init_from_vec(detail::vector_index_sequence<Indices...>,
                     const vec<dataT,N>& other)
  {
    auto dummy_initializer =
    {
      ((_impl.template set<Start_index + Indices>(
                                   other._impl.template get<Indices>())), 0)...
    };
  }

  template<int Start_index, int N>
  __host__ __device__
  constexpr void init_from_vec(const vec<dataT,N>& other)
  {
    init_from_vec<Start_index>(detail::vector_impl<dataT,N>::indices(),
                               other);
  }

  template<int Position>
  __host__ __device__
  constexpr void init_from_argument()
  {
    static_assert (Position == numElements, "Invalid number of vector "
                                            "constructor arguments.");
  }

  template<int Position, typename... Args>
  __host__ __device__
  constexpr void init_from_argument(dataT scalar,
                                    const Args&... args)
  {
    _impl.template set<Position>(scalar);
    init_from_argument<Position+1>(args...);
  }

  template<int Position, int N, typename... Args>
  __host__ __device__
  constexpr void init_from_argument(const vec<dataT,N>& v,
                                    const Args&... args)
  {
    init_from_vec<Position>(v);
    init_from_argument<Position + N>(args...);
  }

public:
  static_assert(numElements == 1 ||
                numElements == 2 ||
                numElements == 3 ||
                numElements == 4 ||
                numElements == 8 ||
                numElements == 16,
                "Invalid number of vector elements. Allowed values: "
                "1,2,3,4,8,16");

  using element_type = dataT;

#ifdef __SYCL_DEVICE_ONLY__
  using vector_t = vec<dataT, numElements>;
#endif
  __host__ __device__
  vec()
    : _impl {}
  {}

  __host__ __device__
  explicit vec(const dataT &arg)
  {
    broadcast_scalar_init(typename detail::vector_impl<dataT,numElements>::indices(),
                          arg);
  }

  template <typename... argTN>
  __host__ __device__
  vec(const argTN&... args)
  {
    init_from_argument<0>(args...);
  }

  vec(const vec<dataT, numElements> &rhs) = default;

#ifdef __SYCL_DEVICE_ONLY__
  vec(vector_t openclVector);
  operator vector_t() const;
#endif

  // Available only when: numElements == 1
  template<int N = numElements,
           std::enable_if_t<N == 1>* = nullptr>
  __host__ __device__
  operator dataT() const
  { return _impl.template get<0>(); }

  __host__ __device__
  size_t get_count() const
  { return numElements; }

  __host__ __device__
  size_t get_size() const
  { return numElements * sizeof (dataT); }

  template <typename convertT, rounding_mode roundingMode>
  vec<convertT, numElements> convert() const;

  template <typename asT>
  asT as() const;

#define HIPSYCL_DEFINE_VECTOR_ACCESS_IF(condition, name, id) \
  template<int N = numElements, \
           std::enable_if_t<(id < N) && (condition)>* = nullptr> \
  __host__ __device__ \
  dataT& name() \
  { return _impl.template get<id>(); } \
  \
  template<int N = numElements, \
           std::enable_if_t<(id < N) && (condition)>* = nullptr> \
  __host__ __device__ \
  dataT name() const \
  { return _impl.template get<id>(); }

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 4, x, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 4, y, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 4, z, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 4, w, 3)

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements == 4, r, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements == 4, g, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements == 4, b, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements == 4, a, 3)

  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s0, 0)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s1, 1)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s2, 2)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s3, 3)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s4, 4)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s5, 5)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s6, 6)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s7, 7)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s8, 8)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, s9, 9)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sA, 10)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sB, 11)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sC, 12)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sD, 13)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sE, 14)
  HIPSYCL_DEFINE_VECTOR_ACCESS_IF(numElements <= 16, sF, 15)


  template<int... swizzleIndexes>
  vec<dataT,numElements> swizzle() const;

  template<int N = numElements,
           std::enable_if_t<(N > 1)>* = nullptr>
  auto lo() const
  { return swizzle<typename detail::vector_impl<dataT, numElements>::lo_indices>(); }

  template<int N = numElements,
           std::enable_if_t<(N > 1)>* = nullptr>
  auto hi() const
  { return swizzle<typename detail::vector_impl<dataT, numElements>::hi_indices>(); }

  template<int N = numElements,
           std::enable_if_t<(N > 1)>* = nullptr>
  auto even() const
  { return swizzle<typename detail::vector_impl<dataT, numElements>::even_indices>(); }

  template<int N = numElements,
           std::enable_if_t<(N > 1)>* = nullptr>
  auto odd() const
  { return swizzle<typename detail::vector_impl<dataT, numElements>::odd_indices>(); }

#ifdef SYCL_SIMPLE_SWIZZLES
#define HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF(condition, name, i0, i1) \
  auto name(std::enable_if_t<(condition)>* = nullptr) const \
  {return swizzle<i0,i1>(); }

#define HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF(condition, name, i0, i1, i2) \
  auto name(std::enable_if_t<(condition)>* = nullptr) const \
  {return swizzle<i0,i1,i2>(); }

#define HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF(condition, name, i0, i1, i2, i3) \
  auto name(std::enable_if_t<(condition)>* = nullptr) const \
  {return swizzle<i0,i1,i2,i3>(); }

  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 1), xx, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), rr, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 2), yx, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), gr, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 3), zx, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), br, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), wx, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ar, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 2), xy, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), rg, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 2), yy, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), gg, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 3), zy, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), bg, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), wy, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ag, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 3), xz, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), rb, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 3), yz, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), gb, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 3), zz, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), bb, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), wz, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ab, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), xw, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ra, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), yw, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ga, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements >= 4), zw, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE2_IF((numElements == 4), ba, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 1), xxx, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rrr, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), yxx, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), grr, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zxx, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), brr, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wxx, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), arr, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), xyx, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rgr, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), yyx, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), ggr, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zyx, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bgr, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wyx, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), agr, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), xzx, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rbr, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), yzx, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gbr, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zzx, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bbr, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wzx, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), abr, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xwx, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rar, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), ywx, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gar, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zwx, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bar, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wwx, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), aar, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), xxy, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rrg, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), yxy, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), grg, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zxy, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), brg, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wxy, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), arg, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), xyy, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rgg, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 2), yyy, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), ggg, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zyy, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bgg, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wyy, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), agg, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), xzy, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rbg, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), yzy, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gbg, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zzy, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bbg, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wzy, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), abg, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xwy, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rag, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), ywy, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gag, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zwy, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bag, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wwy, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), aag, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), xxz, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rrb, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), yxz, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), grb, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zxz, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), brb, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wxz, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), arb, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), xyz, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rgb, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), yyz, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), ggb, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zyz, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bgb, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wyz, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), agb, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), xzz, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rbb, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), yzz, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gbb, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 3), zzz, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bbb, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wzz, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), abb, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xwz, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rab, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), ywz, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gab, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zwz, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bab, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wwz, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), aab, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xxw, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rra, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), yxw, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gra, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zxw, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bra, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wxw, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), ara, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xyw, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rga, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), yyw, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gga, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zyw, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bga, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wyw, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), aga, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xzw, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), rba, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), yzw, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gba, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zzw, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), bba, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), wzw, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), aba, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), xww, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), raa, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), yww, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), gaa, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements >= 4), zww, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE3_IF((numElements == 4), baa, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 1), xxxx, 0, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrrr, 0, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yxxx, 1, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grrr, 1, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxxx, 2, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brrr, 2, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxxx, 3, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arrr, 3, 0, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xyxx, 0, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgrr, 0, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yyxx, 1, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggrr, 1, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyxx, 2, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgrr, 2, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyxx, 3, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agrr, 3, 1, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzxx, 0, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbrr, 0, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzxx, 1, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbrr, 1, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzxx, 2, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbrr, 2, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzxx, 3, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abrr, 3, 2, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwxx, 0, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rarr, 0, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywxx, 1, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), garr, 1, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwxx, 2, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), barr, 2, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwxx, 3, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aarr, 3, 3, 0, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xxyx, 0, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrgr, 0, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yxyx, 1, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grgr, 1, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxyx, 2, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brgr, 2, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxyx, 3, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), argr, 3, 0, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xyyx, 0, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rggr, 0, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yyyx, 1, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gggr, 1, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyyx, 2, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bggr, 2, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyyx, 3, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aggr, 3, 1, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzyx, 0, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbgr, 0, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzyx, 1, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbgr, 1, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzyx, 2, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbgr, 2, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzyx, 3, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abgr, 3, 2, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwyx, 0, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ragr, 0, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywyx, 1, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gagr, 1, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwyx, 2, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bagr, 2, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwyx, 3, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aagr, 3, 3, 1, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xxzx, 0, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrbr, 0, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yxzx, 1, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grbr, 1, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxzx, 2, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brbr, 2, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxzx, 3, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arbr, 3, 0, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xyzx, 0, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgbr, 0, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yyzx, 1, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggbr, 1, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyzx, 2, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgbr, 2, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyzx, 3, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agbr, 3, 1, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzzx, 0, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbbr, 0, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzzx, 1, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbbr, 1, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzzx, 2, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbbr, 2, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzzx, 3, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abbr, 3, 2, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwzx, 0, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rabr, 0, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywzx, 1, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gabr, 1, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwzx, 2, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), babr, 2, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwzx, 3, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aabr, 3, 3, 2, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxwx, 0, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrar, 0, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxwx, 1, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grar, 1, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxwx, 2, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brar, 2, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxwx, 3, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arar, 3, 0, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xywx, 0, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgar, 0, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yywx, 1, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggar, 1, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zywx, 2, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgar, 2, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wywx, 3, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agar, 3, 1, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzwx, 0, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbar, 0, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzwx, 1, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbar, 1, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzwx, 2, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbar, 2, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzwx, 3, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abar, 3, 2, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwwx, 0, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raar, 0, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywwx, 1, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaar, 1, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwwx, 2, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baar, 2, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwwx, 3, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aaar, 3, 3, 3, 0)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xxxy, 0, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrrg, 0, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yxxy, 1, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grrg, 1, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxxy, 2, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brrg, 2, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxxy, 3, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arrg, 3, 0, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xyxy, 0, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgrg, 0, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yyxy, 1, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggrg, 1, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyxy, 2, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgrg, 2, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyxy, 3, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agrg, 3, 1, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzxy, 0, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbrg, 0, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzxy, 1, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbrg, 1, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzxy, 2, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbrg, 2, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzxy, 3, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abrg, 3, 2, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwxy, 0, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rarg, 0, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywxy, 1, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), garg, 1, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwxy, 2, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), barg, 2, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwxy, 3, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aarg, 3, 3, 0, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xxyy, 0, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrgg, 0, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yxyy, 1, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grgg, 1, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxyy, 2, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brgg, 2, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxyy, 3, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), argg, 3, 0, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), xyyy, 0, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rggg, 0, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 2), yyyy, 1, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gggg, 1, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyyy, 2, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bggg, 2, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyyy, 3, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aggg, 3, 1, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzyy, 0, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbgg, 0, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzyy, 1, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbgg, 1, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzyy, 2, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbgg, 2, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzyy, 3, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abgg, 3, 2, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwyy, 0, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ragg, 0, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywyy, 1, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gagg, 1, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwyy, 2, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bagg, 2, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwyy, 3, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aagg, 3, 3, 1, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xxzy, 0, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrbg, 0, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yxzy, 1, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grbg, 1, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxzy, 2, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brbg, 2, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxzy, 3, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arbg, 3, 0, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xyzy, 0, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgbg, 0, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yyzy, 1, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggbg, 1, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyzy, 2, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgbg, 2, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyzy, 3, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agbg, 3, 1, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzzy, 0, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbbg, 0, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzzy, 1, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbbg, 1, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzzy, 2, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbbg, 2, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzzy, 3, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abbg, 3, 2, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwzy, 0, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rabg, 0, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywzy, 1, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gabg, 1, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwzy, 2, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), babg, 2, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwzy, 3, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aabg, 3, 3, 2, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxwy, 0, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrag, 0, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxwy, 1, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grag, 1, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxwy, 2, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brag, 2, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxwy, 3, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arag, 3, 0, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xywy, 0, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgag, 0, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yywy, 1, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggag, 1, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zywy, 2, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgag, 2, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wywy, 3, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agag, 3, 1, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzwy, 0, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbag, 0, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzwy, 1, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbag, 1, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzwy, 2, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbag, 2, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzwy, 3, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abag, 3, 2, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwwy, 0, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raag, 0, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywwy, 1, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaag, 1, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwwy, 2, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baag, 2, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwwy, 3, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aaag, 3, 3, 3, 1)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xxxz, 0, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrrb, 0, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yxxz, 1, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grrb, 1, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxxz, 2, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brrb, 2, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxxz, 3, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arrb, 3, 0, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xyxz, 0, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgrb, 0, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yyxz, 1, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggrb, 1, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyxz, 2, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgrb, 2, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyxz, 3, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agrb, 3, 1, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzxz, 0, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbrb, 0, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzxz, 1, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbrb, 1, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzxz, 2, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbrb, 2, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzxz, 3, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abrb, 3, 2, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwxz, 0, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rarb, 0, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywxz, 1, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), garb, 1, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwxz, 2, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), barb, 2, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwxz, 3, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aarb, 3, 3, 0, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xxyz, 0, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrgb, 0, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yxyz, 1, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grgb, 1, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxyz, 2, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brgb, 2, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxyz, 3, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), argb, 3, 0, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xyyz, 0, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rggb, 0, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yyyz, 1, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gggb, 1, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyyz, 2, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bggb, 2, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyyz, 3, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aggb, 3, 1, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzyz, 0, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbgb, 0, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzyz, 1, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbgb, 1, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzyz, 2, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbgb, 2, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzyz, 3, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abgb, 3, 2, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwyz, 0, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ragb, 0, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywyz, 1, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gagb, 1, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwyz, 2, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bagb, 2, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwyz, 3, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aagb, 3, 3, 1, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xxzz, 0, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrbb, 0, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yxzz, 1, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grbb, 1, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zxzz, 2, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brbb, 2, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxzz, 3, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arbb, 3, 0, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xyzz, 0, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgbb, 0, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yyzz, 1, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggbb, 1, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zyzz, 2, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgbb, 2, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyzz, 3, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agbb, 3, 1, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), xzzz, 0, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbbb, 0, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), yzzz, 1, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbbb, 1, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 3), zzzz, 2, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbbb, 2, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzzz, 3, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abbb, 3, 2, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwzz, 0, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rabb, 0, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywzz, 1, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gabb, 1, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwzz, 2, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), babb, 2, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwzz, 3, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aabb, 3, 3, 2, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxwz, 0, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrab, 0, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxwz, 1, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grab, 1, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxwz, 2, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brab, 2, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxwz, 3, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arab, 3, 0, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xywz, 0, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgab, 0, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yywz, 1, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggab, 1, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zywz, 2, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgab, 2, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wywz, 3, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agab, 3, 1, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzwz, 0, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbab, 0, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzwz, 1, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbab, 1, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzwz, 2, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbab, 2, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzwz, 3, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abab, 3, 2, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwwz, 0, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raab, 0, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywwz, 1, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaab, 1, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwwz, 2, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baab, 2, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwwz, 3, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aaab, 3, 3, 3, 2)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxxw, 0, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrra, 0, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxxw, 1, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grra, 1, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxxw, 2, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brra, 2, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxxw, 3, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arra, 3, 0, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xyxw, 0, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgra, 0, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yyxw, 1, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggra, 1, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zyxw, 2, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgra, 2, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyxw, 3, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agra, 3, 1, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzxw, 0, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbra, 0, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzxw, 1, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbra, 1, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzxw, 2, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbra, 2, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzxw, 3, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abra, 3, 2, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwxw, 0, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rara, 0, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywxw, 1, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gara, 1, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwxw, 2, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bara, 2, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwxw, 3, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aara, 3, 3, 0, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxyw, 0, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrga, 0, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxyw, 1, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grga, 1, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxyw, 2, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brga, 2, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxyw, 3, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arga, 3, 0, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xyyw, 0, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgga, 0, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yyyw, 1, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggga, 1, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zyyw, 2, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgga, 2, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyyw, 3, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agga, 3, 1, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzyw, 0, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbga, 0, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzyw, 1, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbga, 1, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzyw, 2, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbga, 2, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzyw, 3, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abga, 3, 2, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwyw, 0, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raga, 0, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywyw, 1, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaga, 1, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwyw, 2, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baga, 2, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwyw, 3, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aaga, 3, 3, 1, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxzw, 0, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rrba, 0, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxzw, 1, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), grba, 1, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxzw, 2, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), brba, 2, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxzw, 3, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), arba, 3, 0, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xyzw, 0, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgba, 0, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yyzw, 1, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggba, 1, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zyzw, 2, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgba, 2, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyzw, 3, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agba, 3, 1, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzzw, 0, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbba, 0, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzzw, 1, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbba, 1, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzzw, 2, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbba, 2, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzzw, 3, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abba, 3, 2, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwzw, 0, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raba, 0, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywzw, 1, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaba, 1, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwzw, 2, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baba, 2, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wwzw, 3, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), aaba, 3, 3, 2, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xxww, 0, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rraa, 0, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yxww, 1, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), graa, 1, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zxww, 2, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), braa, 2, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wxww, 3, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), araa, 3, 0, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xyww, 0, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rgaa, 0, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yyww, 1, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), ggaa, 1, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zyww, 2, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bgaa, 2, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wyww, 3, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), agaa, 3, 1, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xzww, 0, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), rbaa, 0, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), yzww, 1, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gbaa, 1, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zzww, 2, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), bbaa, 2, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), wzww, 3, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), abaa, 3, 2, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), xwww, 0, 3, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), raaa, 0, 3, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), ywww, 1, 3, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), gaaa, 1, 3, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements >= 4), zwww, 2, 3, 3, 3)
  HIPSYCL_DEFINE_VECTOR_SWIZZLE4_IF((numElements == 4), baaa, 2, 3, 3, 3)

#endif


  // load and store member functions
  template <access::address_space addressSpace>
  void load(size_t offset, multi_ptr<dataT, addressSpace> ptr);
  template <access::address_space addressSpace>
  void store(size_t offset, multi_ptr<dataT, addressSpace> ptr) const;

  // OP is: +, -, *, /, %
  /* When OP is % available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  __host__ __device__
  vec<dataT, numElements> operator+(const vec<dataT, numElements> &rhs) const
  { return vec<dataT,numElements>{_impl + rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator-(const vec<dataT, numElements> &rhs) const
  { return vec<dataT,numElements>{_impl - rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator*(const vec<dataT, numElements> &rhs) const
  { return vec<dataT,numElements>{_impl * rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator/(const vec<dataT, numElements> &rhs) const
  { return vec<dataT,numElements>{_impl / rhs._impl}; }

  template<class t = dataT,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  __host__ __device__
  vec<dataT, numElements> operator%(const vec<dataT, numElements> &rhs) const
  { return vec<dataT,numElements>{_impl % rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator+(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl + rhs}; }

  __host__ __device__
  vec<dataT, numElements> operator-(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl - rhs}; }

  __host__ __device__
  vec<dataT, numElements> operator*(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl * rhs}; }

  __host__ __device__
  vec<dataT, numElements> operator/(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl / rhs}; }


  // OP is: +=, -=, *=, /=, %=
  /* When OP is %= available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  __host__ __device__
  vec<dataT, numElements> &operator+=(const vec<dataT, numElements> &rhs)
  { _impl += rhs._impl; return *this; }

  __host__ __device__
  vec<dataT, numElements> &operator-=(const vec<dataT, numElements> &rhs)
  { _impl -= rhs._impl; return *this; }

  __host__ __device__
  vec<dataT, numElements> &operator*=(const vec<dataT, numElements> &rhs)
  { _impl *= rhs._impl; return *this; }

  __host__ __device__
  vec<dataT, numElements> &operator/=(const vec<dataT, numElements> &rhs)
  { _impl /= rhs._impl; return *this; }


  template<class t = dataT,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  __host__ __device__
  vec<dataT, numElements> &operator%=(const vec<dataT, numElements> &rhs)
  { _impl %= rhs._impl; return *this; }

  // ToDo
  vec<dataT, numElements> &operatorOP(const dataT &rhs);

  // OP is: ++, --
  __host__ __device__
  vec<dataT, numElements> &operator++()
  { *this += 1; return *this; }

  __host__ __device__
  vec<dataT, numElements> &operator--()
  { *this -= 1; return *this; }

  __host__ __device__
  vec<dataT, numElements> operator++(int)
  {
    vec<dataT, numElements> old = *this;
    ++(*this);
    return old;
  }

  __host__ __device__
  vec<dataT, numElements> operator--(int)
  {
    vec<dataT, numElements> old = *this;
    --(*this);
    return old;
  }

  // OP is: &, |, ˆ
  /* Available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  //vec<dataT, numElements> operatorOP(const vec<dataT, numElements> &rhs) const;
  //vec<dataT, numElements> operatorOP(const dataT &rhs) const;

  __host__ __device__
  vec<dataT, numElements> operator&(const vec<dataT,numElements> &rhs) const
  { return vec<dataT,numElements>{_impl & rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator|(const vec<dataT,numElements> &rhs) const
  { return vec<dataT,numElements>{_impl | rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator^(const vec<dataT,numElements> &rhs) const
  { return vec<dataT,numElements>{_impl ^ rhs._impl}; }

  __host__ __device__
  vec<dataT, numElements> operator&(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl & rhs}; }

  __host__ __device__
  vec<dataT, numElements> operator|(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl | rhs}; }

  __host__ __device__
  vec<dataT, numElements> operator^(const dataT &rhs) const
  { return vec<dataT,numElements>{_impl ^ rhs}; }

  // OP is: &=, |=, ˆ=
  /* Available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  // vec<dataT, numElements> &operatorOP(const vec<dataT, numElements> &rhs);
  // vec<dataT, numElements> &operatorOP(const dataT &rhs);

  // OP is: &&, ||
  // vec<RET, numElements> operatorOP(const vec<dataT, numElements> &rhs) const;
  // vec<RET, numElements> operatorOP(const dataT &rhs) const;

  // OP is: <<, >>
  /* Available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  // vec<dataT, numElements> operatorOP(const vec<dataT, numElements> &rhs) const;
  // vec<dataT, numElements> operatorOP(const dataT &rhs) const;

  // OP is: <<=, >>=
  /* Available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  // vec<dataT, numElements> &operatorOP(const vec<dataT, numElements> &rhs);
  // vec<dataT, numElements> &operatorOP(const dataT &rhs);

  // OP is: ==, !=, <, >, <=, >=
  //vec<RET, numElements> operatorOP(const vec<dataT, numElements> &rhs) const;
  //vec<RET, numElements> operatorOP(const dataT &rhs) const;
  //vec<dataT, numElements> &operator=(const vec<dataT, numElements> &rhs);
  //vec<dataT, numElements> &operator=(const dataT &rhs);

  /* Available only when: dataT != cl_float && dataT != cl_double
&& dataT != cl_half. */
  //vec<dataT, numElements> operator~();
  //vec<RET, numElements> operator!();
  //vec<dataT, numElements> &operator=(const vec<dataT, numElements> &rhs);
  //vec<dataT, numElements> &operator=(const dataT &rhs);

private:
  detail::vector_impl<dataT, numElements> _impl;
};

/*

// OP is: +, -, *, /, %
template <typename dataT, int numElements>
vec<dataT, numElements> operatorOP(const dataT &lhs,
                                   const vec<dataT, numElements> &rhs);
// OP is: &, |, ˆ
// Available only when: dataT != cl_float && dataT != cl_double && dataT != cl_half.
template <typename dataT, int numElements>
vec<dataT, numElements> operatorOP(const dataT &lhs,
                                   const vec<dataT, numElements> &rhs);
// OP is: &&, ||
template <typename dataT, int numElements>
vec<RET, numElements> operatorOP(const dataT &lhs,
                                 const vec<dataT, numElements> &rhs);
// OP is: <<, >>
// Available only when: dataT != cl_float && dataT != cl_double && dataT != cl_half.
template <typename dataT, int numElements>
vec<dataT, numElements> operatorOP(const dataT &lhs,
                                   const vec<dataT, numElements> &rhs);
// OP is: ==, !=, <, >, <=, >=
template <typename dataT, int numElements>
vec<RET, numElements> operatorOP(const dataT &lhs,
                                 const vec<dataT, numElements> &rhs);

*/

#define HIPSYCL_DEFINE_VECTOR_ALIAS(T, alias) \
  using alias ## 2  = vec<T, 2 >; \
  using alias ## 3  = vec<T, 3 >; \
  using alias ## 4  = vec<T, 4 >; \
  using alias ## 8  = vec<T, 8 >; \
  using alias ## 16 = vec<T, 16>

HIPSYCL_DEFINE_VECTOR_ALIAS(char, char);
HIPSYCL_DEFINE_VECTOR_ALIAS(short, short);
HIPSYCL_DEFINE_VECTOR_ALIAS(int, int);
HIPSYCL_DEFINE_VECTOR_ALIAS(long, long);
HIPSYCL_DEFINE_VECTOR_ALIAS(float, float);
HIPSYCL_DEFINE_VECTOR_ALIAS(double, double);
// ToDo: half
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_char, cl_char);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_uchar, cl_uchar);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_short, cl_short);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_ushort, cl_ushort);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_int, cl_int);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_uint, cl_uint);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_long, cl_long);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_ulong, cl_ulong);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_float, cl_float);
HIPSYCL_DEFINE_VECTOR_ALIAS(cl_double, cl_double);
// ToDo: half
HIPSYCL_DEFINE_VECTOR_ALIAS(signed char, schar);
HIPSYCL_DEFINE_VECTOR_ALIAS(unsigned char, uchar);
HIPSYCL_DEFINE_VECTOR_ALIAS(unsigned short, ushort);
HIPSYCL_DEFINE_VECTOR_ALIAS(unsigned int, uint);
HIPSYCL_DEFINE_VECTOR_ALIAS(unsigned long, ulong);
HIPSYCL_DEFINE_VECTOR_ALIAS(long long, longlong);
HIPSYCL_DEFINE_VECTOR_ALIAS(unsigned long long, ulonglong);


namespace detail {

template<class T, int N, class Function>
__host__ __device__
void transform_vector(vec<T,N>& v, Function f)
{
  v._impl.transform(f);
}

template<class T, int N, class Function>
__host__ __device__
vec<T,N> binary_vector_operation(const vec<T,N>& a,
                                 const vec<T,N>& b,
                                 Function f)
{
  return a._impl.binary_operation(f, b);
}

template<class T, int N, class Function>
__host__ __device__
vec<T,N> trinary_vector_operation(const vec<T,N>& a,
                                  const vec<T,N>& b,
                                  const vec<T,N>& c,
                                  Function f)
{
  return vec<T,N>{
        vector_impl<T,N>::trinary_operation(f, a._impl, b._impl, c._impl)
  };
}

} // detail
} // namespace sycl
} // namespace cl


#endif
