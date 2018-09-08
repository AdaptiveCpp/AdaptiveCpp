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


namespace cl {
namespace sycl {

template<typename dataT, int N>
class vec;

namespace detail {

template<class T, int N>
struct intrinsic_vector
{
  static constexpr bool exists = false;
};

#define HIPSYCL_DEFINE_INTRINSIC_VECTOR( \
  T,                  \
  num_elements,       \
  mapped_vector_type) \
template<> struct intrinsic_vector<T,num_elements> \
{ \
  using type = mapped_vector_type;  \
  static constexpr bool exists = true; \
}

HIPSYCL_DEFINE_INTRINSIC_VECTOR(char, 1, ::char1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(char, 2, ::char2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(char, 3, ::char3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(char, 4, ::char4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(bool, 1, ::char1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(bool, 2, ::char2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(bool, 3, ::char3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(bool, 4, ::char4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned char, 1, ::uchar1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned char, 2, ::uchar2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned char, 3, ::uchar3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned char, 4, ::uchar4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(short, 1, ::short1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(short, 2, ::short2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(short, 3, ::short3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(short, 4, ::short4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned short, 1, ::ushort1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned short, 2, ::ushort2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned short, 3, ::ushort3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned short, 4, ::ushort4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(int, 1, ::int1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(int, 2, ::int2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(int, 3, ::int3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(int, 4, ::int4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned int, 1, ::uint1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned int, 2, ::uint2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned int, 3, ::uint3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned int, 4, ::uint4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(long, 1, ::long1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long, 2, ::long2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long, 3, ::long3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long, 4, ::long4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long, 1, ::ulong1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long, 2, ::ulong2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long, 3, ::ulong3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long, 4, ::ulong4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(float, 1, ::float1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(float, 2, ::float2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(float, 3, ::float3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(float, 4, ::float4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(long long, 1, ::longlong1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long long, 2, ::longlong2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long long, 3, ::longlong3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(long long, 4, ::longlong4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long long, 1, ::ulonglong1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long long, 2, ::ulonglong2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long long, 3, ::ulonglong3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(unsigned long long, 4, ::ulonglong4);

HIPSYCL_DEFINE_INTRINSIC_VECTOR(double, 1, ::double1);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(double, 2, ::double2);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(double, 3, ::double3);
HIPSYCL_DEFINE_INTRINSIC_VECTOR(double, 4, ::double4);

template<class T, int N, int Index>
struct vector_accessor
{};

#define HIPSYCL_DEFINE_VECTOR_ACCESSOR(index, element_name) \
template<class T, int N>                 \
struct vector_accessor<T,N,index> \
{ \
  __host__ __device__ \
  static T& get(typename intrinsic_vector<T,N>::type& v) \
  { return v.element_name; } \
\
  __host__ __device__ \
  static T get(const typename intrinsic_vector<T,N>::type& v) \
  { return v.element_name; } \
}

HIPSYCL_DEFINE_VECTOR_ACCESSOR(0, x);
HIPSYCL_DEFINE_VECTOR_ACCESSOR(1, y);
HIPSYCL_DEFINE_VECTOR_ACCESSOR(2, z);
HIPSYCL_DEFINE_VECTOR_ACCESSOR(3, w);

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP1(lhs, rhs, op) \
  lhs.data.x op rhs.data.x

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP2(lhs, rhs, op) \
  lhs.data.x op rhs.data.x; \
  lhs.data.y op rhs.data.y

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP3(lhs, rhs, op) \
  lhs.data.x op rhs.data.x; \
  lhs.data.y op rhs.data.y; \
  lhs.data.z op rhs.data.z

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP4(lhs, rhs, op) \
  lhs.data.x op rhs.data.x; \
  lhs.data.y op rhs.data.y; \
  lhs.data.z op rhs.data.z; \
  lhs.data.w op rhs.data.w

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP8(lhs, rhs, op) \
  lhs.data0.x op rhs.data0.x; \
  lhs.data0.y op rhs.data0.y; \
  lhs.data0.z op rhs.data0.z; \
  lhs.data0.w op rhs.data0.w; \
  lhs.data1.x op rhs.data1.x; \
  lhs.data1.y op rhs.data1.y; \
  lhs.data1.z op rhs.data1.z; \
  lhs.data1.w op rhs.data1.w

#define HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP16(lhs, rhs, op) \
  lhs.data0.x op rhs.data0.x; \
  lhs.data0.y op rhs.data0.y; \
  lhs.data0.z op rhs.data0.z; \
  lhs.data0.w op rhs.data0.w; \
  lhs.data1.x op rhs.data1.x; \
  lhs.data1.y op rhs.data1.y; \
  lhs.data1.z op rhs.data1.z; \
  lhs.data1.w op rhs.data1.w; \
  lhs.data2.x op rhs.data2.x; \
  lhs.data2.y op rhs.data2.y; \
  lhs.data2.z op rhs.data2.z; \
  lhs.data2.w op rhs.data2.w; \
  lhs.data3.x op rhs.data3.x; \
  lhs.data3.y op rhs.data3.y; \
  lhs.data3.z op rhs.data3.z; \
  lhs.data3.w op rhs.data3.w

#define HIPSYCL_BINARY_COMPONENTWISE_OP1(result, lhs, rhs, op) \
  result.data.x = lhs.data.x op rhs.data.x

#define HIPSYCL_BINARY_COMPONENTWISE_OP2(result, lhs, rhs, op) \
  result.data.x = lhs.data.x op rhs.data.x; \
  result.data.y = lhs.data.y op rhs.data.y

#define HIPSYCL_BINARY_COMPONENTWISE_OP3(result, lhs, rhs, op) \
  result.data.x = lhs.data.x op rhs.data.x; \
  result.data.y = lhs.data.y op rhs.data.y; \
  result.data.z = lhs.data.z op rhs.data.z

#define HIPSYCL_BINARY_COMPONENTWISE_OP4(result, lhs, rhs, op) \
  result.data.x = lhs.data.x op rhs.data.x; \
  result.data.y = lhs.data.y op rhs.data.y; \
  result.data.z = lhs.data.z op rhs.data.z; \
  result.data.w = lhs.data.w op rhs.data.w

#define HIPSYCL_BINARY_COMPONENTWISE_OP8(result, lhs, rhs, op) \
  result.data0.x = lhs.data0.x op rhs.data0.x; \
  result.data0.y = lhs.data0.y op rhs.data0.y; \
  result.data0.z = lhs.data0.z op rhs.data0.z; \
  result.data0.w = lhs.data0.w op rhs.data0.w; \
  result.data1.x = lhs.data1.x op rhs.data1.x; \
  result.data1.y = lhs.data1.y op rhs.data1.y; \
  result.data1.z = lhs.data1.z op rhs.data1.z; \
  result.data1.w = lhs.data1.w op rhs.data1.w

#define HIPSYCL_BINARY_COMPONENTWISE_OP16(result, lhs, rhs, op) \
  result.data0.x = lhs.data0.x op rhs.data0.x; \
  result.data0.y = lhs.data0.y op rhs.data0.y; \
  result.data0.z = lhs.data0.z op rhs.data0.z; \
  result.data0.w = lhs.data0.w op rhs.data0.w; \
  result.data1.x = lhs.data1.x op rhs.data1.x; \
  result.data1.y = lhs.data1.y op rhs.data1.y; \
  result.data1.z = lhs.data1.z op rhs.data1.z; \
  result.data1.w = lhs.data1.w op rhs.data1.w; \
  result.data2.x = lhs.data2.x op rhs.data2.x; \
  result.data2.y = lhs.data2.y op rhs.data2.y; \
  result.data2.z = lhs.data2.z op rhs.data2.z; \
  result.data2.w = lhs.data2.w op rhs.data2.w; \
  result.data3.x = lhs.data3.x op rhs.data3.x; \
  result.data3.y = lhs.data3.y op rhs.data3.y; \
  result.data3.z = lhs.data3.z op rhs.data3.z; \
  result.data3.w = lhs.data3.w op rhs.data3.w

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP1(result, lhs, scalar, op) \
  result.data.x = lhs.data.x op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP2(result, lhs, scalar, op) \
  result.data.x = lhs.data.x op scalar; \
  result.data.y = lhs.data.y op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP3(result, lhs, scalar, op) \
  result.data.x = lhs.data.x op scalar; \
  result.data.y = lhs.data.y op scalar; \
  result.data.z = lhs.data.z op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP4(result, lhs, scalar, op) \
  result.data.x = lhs.data.x op scalar; \
  result.data.y = lhs.data.y op scalar; \
  result.data.z = lhs.data.z op scalar; \
  result.data.w = lhs.data.w op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP8(result, lhs, scalar, op) \
  result.data0.x = lhs.data0.x op scalar; \
  result.data0.y = lhs.data0.y op scalar; \
  result.data0.z = lhs.data0.z op scalar; \
  result.data0.w = lhs.data0.w op scalar; \
  result.data1.x = lhs.data1.x op scalar; \
  result.data1.y = lhs.data1.y op scalar; \
  result.data1.z = lhs.data1.z op scalar; \
  result.data1.w = lhs.data1.w op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP16(result, lhs, scalar, op) \
  result.data0.x = lhs.data0.x op scalar; \
  result.data0.y = lhs.data0.y op scalar; \
  result.data0.z = lhs.data0.z op scalar; \
  result.data0.w = lhs.data0.w op scalar; \
  result.data1.x = lhs.data1.x op scalar; \
  result.data1.y = lhs.data1.y op scalar; \
  result.data1.z = lhs.data1.z op scalar; \
  result.data1.w = lhs.data1.w op scalar; \
  result.data2.x = lhs.data2.x op scalar; \
  result.data2.y = lhs.data2.y op scalar; \
  result.data2.z = lhs.data2.z op scalar; \
  result.data2.w = lhs.data2.w op scalar; \
  result.data3.x = lhs.data3.x op scalar; \
  result.data3.y = lhs.data3.y op scalar; \
  result.data3.z = lhs.data3.z op scalar; \
  result.data3.w = lhs.data3.w op scalar


#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP1(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP2(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP3(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP4(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP8(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(op) \
__host__ __device__ \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP16(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP1(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP2(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP3(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP4(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP8(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T,op) \
__host__ __device__ \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP16(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP1((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP2((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP3((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP4((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP8((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(op) \
__host__ __device__ \
vector_impl& operator op(const T& rhs) const { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP16((*this), rhs, op); \
  return *this; \
}

template<class T, int N>
struct vector_impl
{
};

template<class T>
struct vector_impl<T,1>
{
  static_assert (intrinsic_vector<T,1>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  __host__ __device__
  T& get()
  {
    static_assert (component < 1, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,1,component>::get(data);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    static_assert (component < 1, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,1,component>::get(data);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data.x = f(data.x);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data.x = f(a.data.x,b.data.x,c.data.x);
    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(%=)


  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T,^)

  typename intrinsic_vector<T,1>::type data;
};

template<class T>
struct vector_impl<T,2>
{
  static_assert (intrinsic_vector<T,2>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  __host__ __device__
  T& get()
  {
    static_assert (component < 2, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,2,component>::get(data);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    static_assert (component < 2, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,2,component>::get(data);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    result.data.y = f(data.y, other.data.y);
    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data.x = f(a.data.x, b.data.x, c.data.x);
    result.data.y = f(a.data.y, b.data.y, c.data.y);
    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(%=)

  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T,^)

  typename intrinsic_vector<T,2>::type data;
};

template<class T>
struct vector_impl<T,3>
{
  static_assert (intrinsic_vector<T,3>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  __host__ __device__
  T& get()
  {
    static_assert (component < 3, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,3,component>::get(data);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    static_assert (component < 3, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,3,component>::get(data);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
    data.z = f(data.z);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    result.data.y = f(data.y, other.data.y);
    result.data.z = f(data.z, other.data.z);
    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data.x = f(a.data.x, b.data.x, c.data.x);
    result.data.y = f(a.data.y, b.data.y, c.data.y);
    result.data.z = f(a.data.z, b.data.z, c.data.z);
    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(%=)

  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T,^)

  typename intrinsic_vector<T,3>::type data;
};

template<class T>
struct vector_impl<T,4>
{
  static_assert (intrinsic_vector<T,4>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  __host__ __device__
  T& get()
  {
    static_assert (component < 4, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,4,component>::get(data);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    static_assert (component < 4, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,4,component>::get(data);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
    data.z = f(data.z);
    data.w = f(data.w);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    result.data.y = f(data.y, other.data.y);
    result.data.z = f(data.z, other.data.z);
    result.data.w = f(data.w, other.data.w);
    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data.x = f(a.data.x, b.data.x, c.data.x);
    result.data.y = f(a.data.y, b.data.y, c.data.y);
    result.data.z = f(a.data.z, b.data.z, c.data.z);
    result.data.w = f(a.data.w, b.data.w, c.data.w);
    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(%=)

  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T,^)

  typename intrinsic_vector<T,4>::type data;
};


template<class T, int Index>
struct vector_multi_accessor
{};

#define HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(Index, value) \
template<class T> \
struct vector_multi_accessor<T,Index> \
{ \
  template<class Vector_type>   \
  __host__ __device__           \
  static T& get(Vector_type& v) \
  { return v.value; }           \
                                \
  template<class Vector_type>   \
  __host__ __device__           \
  static T get(const Vector_type& v) \
  { return v.value; } \
}

HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(0, data0.x);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(1, data0.y);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(2, data0.z);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(3, data0.w);

HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(4, data1.x);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(5, data1.y);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(6, data1.z);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(7, data1.w);

HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(8,  data2.x);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(9,  data2.y);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(10, data2.z);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(11, data2.w);

HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(12, data3.x);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(13, data3.y);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(14, data3.z);
HIPSYCL_DEFINE_VECTOR_MULTI_ACCESSOR(15, data3.w);

template<class T>
struct vector_impl<T,8>
{

  template<int component>
  __host__ __device__
  T& get()
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data0.x = f(data0.x);
    data0.y = f(data0.y);
    data0.z = f(data0.z);
    data0.w = f(data0.w);

    data1.x = f(data1.x);
    data1.y = f(data1.y);
    data1.z = f(data1.z);
    data1.w = f(data1.w);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data0.x = f(data0.x, other.data0.x);
    result.data0.y = f(data0.y, other.data0.y);
    result.data0.z = f(data0.z, other.data0.z);
    result.data0.w = f(data0.w, other.data0.w);

    result.data1.x = f(data1.x, other.data1.x);
    result.data1.y = f(data1.y, other.data1.y);
    result.data1.z = f(data1.z, other.data1.z);
    result.data1.w = f(data1.w, other.data1.w);

    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data0.x = f(a.data0.x, b.data0.x, c.data0.x);
    result.data0.y = f(a.data0.y, b.data0.y, c.data0.y);
    result.data0.z = f(a.data0.z, b.data0.z, c.data0.z);
    result.data0.w = f(a.data0.w, b.data0.w, c.data0.w);

    result.data1.x = f(a.data1.x, b.data1.x, c.data1.x);
    result.data1.y = f(a.data1.y, b.data1.y, c.data1.y);
    result.data1.z = f(a.data1.z, b.data1.z, c.data1.z);
    result.data1.w = f(a.data1.w, b.data1.w, c.data1.w);

    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(%=)

  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T,^)

  typename intrinsic_vector<T,4>::type data0;
  typename intrinsic_vector<T,4>::type data1;
};

template<class T>
struct vector_impl<T,16>
{
  template<int component>
  __host__ __device__
  T& get()
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  __host__ __device__
  T get() const
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<class Function>
  __host__ __device__
  void transform(Function f)
  {
    data0.x = f(data0.x);
    data0.y = f(data0.y);
    data0.z = f(data0.z);
    data0.w = f(data0.w);

    data1.x = f(data1.x);
    data1.y = f(data1.y);
    data1.z = f(data1.z);
    data1.w = f(data1.w);

    data2.x = f(data2.x);
    data2.y = f(data2.y);
    data2.z = f(data2.z);
    data2.w = f(data2.w);

    data3.x = f(data3.x);
    data3.y = f(data3.y);
    data3.z = f(data3.z);
    data3.w = f(data3.w);
  }

  template<class Function>
  __host__ __device__
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data0.x = f(data0.x, other.data0.x);
    result.data0.y = f(data0.y, other.data0.y);
    result.data0.z = f(data0.z, other.data0.z);
    result.data0.w = f(data0.w, other.data0.w);

    result.data1.x = f(data1.x, other.data1.x);
    result.data1.y = f(data1.y, other.data1.y);
    result.data1.z = f(data1.z, other.data1.z);
    result.data1.w = f(data1.w, other.data1.w);

    result.data2.x = f(data2.x, other.data2.x);
    result.data2.y = f(data2.y, other.data2.y);
    result.data2.z = f(data2.z, other.data2.z);
    result.data2.w = f(data2.w, other.data2.w);

    result.data3.x = f(data3.x, other.data3.x);
    result.data3.y = f(data3.y, other.data3.y);
    result.data3.z = f(data3.z, other.data3.z);
    result.data3.w = f(data3.w, other.data3.w);

    return result;
  }

  template<class Function>
  __host__ __device__
  static vector_impl trinary_operation(Function f,
                                       const vector_impl& a,
                                       const vector_impl& b,
                                       const vector_impl& c)
  {
    vector_impl result;
    result.data0.x = f(a.data0.x, b.data0.x, c.data0.x);
    result.data0.y = f(a.data0.y, b.data0.y, c.data0.y);
    result.data0.z = f(a.data0.z, b.data0.z, c.data0.z);
    result.data0.w = f(a.data0.w, b.data0.w, c.data0.w);

    result.data1.x = f(a.data1.x, b.data1.x, c.data1.x);
    result.data1.y = f(a.data1.y, b.data1.y, c.data1.y);
    result.data1.z = f(a.data1.z, b.data1.z, c.data1.z);
    result.data1.w = f(a.data1.w, b.data1.w, c.data1.w);

    result.data2.x = f(a.data2.x, b.data2.x, c.data2.x);
    result.data2.y = f(a.data2.y, b.data2.y, c.data2.y);
    result.data2.z = f(a.data2.z, b.data2.z, c.data2.z);
    result.data2.w = f(a.data2.w, b.data2.w, c.data2.w);

    result.data3.x = f(a.data3.x, b.data3.x, c.data3.x);
    result.data3.y = f(a.data3.y, b.data3.y, c.data3.y);
    result.data3.z = f(a.data3.z, b.data3.z, c.data3.z);
    result.data3.w = f(a.data3.w, b.data3.w, c.data3.w);

    return result;
  }

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(+)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(-)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(*)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(/)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(%)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, +)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, -)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, *)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, /)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, %)

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(%=)

  // Bitwise operators
  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(^)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T,&)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T,|)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T,^)


  typename intrinsic_vector<T,4>::type data0;
  typename intrinsic_vector<T,4>::type data1;
  typename intrinsic_vector<T,4>::type data2;
  typename intrinsic_vector<T,4>::type data3;
};


template<class T, int N, class Function>
__host__ __device__
void transform_vector(vec<T,N>& v, Function);

template<class T, int N, class Function>
__host__ __device__
vec<T,N> binary_vector_operation(const vec<T,N>& a,
                                 const vec<T,N>& b,
                                 Function f);

template<class T, int N, class Function>
__host__ __device__
vec<T,N> trinary_vector_operation(const vec<T,N>& a,
                                  const vec<T,N>& b,
                                  const vec<T,N>& c,
                                  Function f);

}


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
  template<class Function>
  __host__ __device__
  friend void detail::transform_vector(vec<dataT,numElements>& v,
                                       Function f);

  template<class T, int N, class Function>
  __host__ __device__
  friend vec<T,N> binary_vector_operation(const vec<T,N>& a,
                                          const vec<T,N>& b,
                                          Function f);

  template<class T, int N, class Function>
  __host__ __device__
  friend vec<T,N> trinary_vector_operation(const vec<T,N>& a,
                                           const vec<T,N>& b,
                                           const vec<T,N>& c,
                                           Function f);

  explicit vec(const detail::vector_impl<dataT,numElements>& v)
    : _impl{v}
  {}
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

  explicit vec(const dataT &arg);

  template <typename... argTN,
            std::enable_if_t<numElements == sizeof...(argTN)>* = nullptr>
  vec(const argTN&... args);

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

  /* TODO: Implement swizzles

  template<int... swizzleIndexes>
  __swizzled_vec__ swizzle() const;

  // Available only when numElements <= 4.
  // XYZW_ACCESS is: x, y, z, w, subject to numElements.
  __swizzled_vec__ XYZW_ACCESS() const
  // Available only numElements == 4.
  // RGBA_ACCESS is: r, g, b, a.
  __swizzled_vec__ RGBA_ACCESS() const
  // INDEX_ACCESS is: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, sA, sB, sC, sD,
  // sE, sF, subject to numElements.
  __swizzled_vec__ INDEX_ACCESS() const
#ifdef SYCL_SIMPLE_SWIZZLES
  // Available only when numElements <= 4.
  // XYZW_SWIZZLE is all permutations with repetition of: x, y, z, w, subject to
  // numElements.
  __swizzled_vec__ XYZW_SWIZZLE() const;
  // Available only when numElements == 4.
  // RGBA_SWIZZLE is all permutations with repetition of: r, g, b, a.
  __swizzled_vec__ RGBA_SWIZZLE() const;
#endif
  // #ifdef SYCL_SIMPLE_SWIZZLES
  // Available only when: numElements > 1.
  __swizzled_vec__ lo() const;
  __swizzled_vec__ hi() const;
  __swizzled_vec__ odd() const;
  __swizzled_vec__ even() const;
  */

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
inline void transform_vector(vec<T,N>& v, Function f)
{
  v._impl.transform(f);
}

template<class T, int N, class Function>
__host__ __device__
inline vec<T,N> binary_vector_operation(const vec<T,N>& a,
                                        const vec<T,N>& b,
                                        Function f)
{
  return a._impl.binary_operation(f, b);
}

template<class T, int N, class Function>
__host__ __device__
inline vec<T,N> trinary_vector_operation(const vec<T,N>& a,
                                         const vec<T,N>& b,
                                         const vec<T,N>& c,
                                         Function f)
{
  return vector_impl<T,N>::trinary_operation(f, a._impl, b._impl, c._impl);
}

} // detail
} // namespace sycl
} // namespace cl


#endif
