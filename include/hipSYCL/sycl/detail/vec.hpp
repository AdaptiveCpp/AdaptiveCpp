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

#ifndef HIPSYCL_DETAIL_VEC_HPP
#define HIPSYCL_DETAIL_VEC_HPP

#include <cstddef>
#include <type_traits>
#include "../backend/backend.hpp"
#include "../types.hpp"


namespace hipsycl {
namespace sycl {

template<typename dataT, int N>
class vec;

namespace detail {

template<class T>
struct logical_vector_op_result
{};

template<> struct logical_vector_op_result<detail::s_char>
{ using type = detail::s_char; };

template<> struct logical_vector_op_result<detail::u_char>
{ using type = detail::s_char; };

template<> struct logical_vector_op_result<detail::s_short>
{ using type = detail::s_short; };

template<> struct logical_vector_op_result<detail::u_short>
{ using type = detail::s_short; };

template<> struct logical_vector_op_result<detail::s_int>
{ using type = detail::s_int; };

template<> struct logical_vector_op_result<detail::u_int>
{ using type = detail::s_int; };

template<> struct logical_vector_op_result<detail::sp_float>
{ using type = detail::s_int; };

template<> struct logical_vector_op_result<detail::s_long>
{ using type = detail::s_long; };

template<> struct logical_vector_op_result<detail::u_long>
{ using type = detail::s_long; };

template<> struct logical_vector_op_result<detail::dp_float>
{ using type = detail::s_long; };

template<int ...>
struct vector_index_sequence { };

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
  HIPSYCL_UNIVERSAL_TARGET \
  static T& get(typename intrinsic_vector<T,N>::type& v) \
  { return v.element_name; } \
\
  HIPSYCL_UNIVERSAL_TARGET \
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

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP1(lhs, scalar, op) \
  lhs.data.x op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP2(lhs, scalar, op) \
  lhs.data.x op scalar; \
  lhs.data.y op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP3(lhs, scalar, op) \
  lhs.data.x op scalar; \
  lhs.data.y op scalar; \
  lhs.data.z op scalar;

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP4(lhs, scalar, op) \
  lhs.data.x op scalar; \
  lhs.data.y op scalar; \
  lhs.data.z op scalar; \
  lhs.data.w op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP8(lhs, scalar, op) \
  lhs.data0.x op scalar; \
  lhs.data0.y op scalar; \
  lhs.data0.z op scalar; \
  lhs.data0.w op scalar; \
  lhs.data1.x op scalar; \
  lhs.data1.y op scalar; \
  lhs.data1.z op scalar; \
  lhs.data1.w op scalar

#define HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP16(lhs, scalar, op) \
  lhs.data0.x op scalar; \
  lhs.data0.y op scalar; \
  lhs.data0.z op scalar; \
  lhs.data0.w op scalar; \
  lhs.data1.x op scalar; \
  lhs.data1.y op scalar; \
  lhs.data1.z op scalar; \
  lhs.data1.w op scalar; \
  lhs.data2.x op scalar; \
  lhs.data2.y op scalar; \
  lhs.data2.z op scalar; \
  lhs.data2.w op scalar; \
  lhs.data3.x op scalar; \
  lhs.data3.y op scalar; \
  lhs.data3.z op scalar; \
  lhs.data3.w op scalar

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP1(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP2(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP3(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP4(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP8(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const vector_impl& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_OP16(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP1(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP2(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP3(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP4(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP8(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T,op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl operator op(const T& rhs) const { \
  vector_impl result; \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_OP16(result, (*this), rhs, op); \
  return result; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP1((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP2((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP3((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP4((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP8((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const vector_impl& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_INPLACE_OP16((*this), rhs, op); \
  return *this; \
}


#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP1((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP2((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP3((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP4((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP1((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(op) \
HIPSYCL_UNIVERSAL_TARGET \
vector_impl& operator op(const T& rhs) { \
  HIPSYCL_BINARY_COMPONENTWISE_VECTOR_SCALAR_INPLACE_OP16((*this), rhs, op); \
  return *this; \
}

#define HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(helper_name, op) \
private: \
template<int... Indices> \
HIPSYCL_UNIVERSAL_TARGET \
auto helper_name(vector_index_sequence<Indices...>, \
                 const vector_impl& rhs) const { \
  using result_type = typename logical_vector_op_result<T>::type; \
  vector_impl<result_type, dimension> result; \
  auto dummy_initializer = { \
    ((result.template set<Indices>(static_cast<result_type>(get<Indices>() \
                          op rhs.get<Indices>()))), 0)... \
  }; \
  return result; \
} \
public: \
HIPSYCL_UNIVERSAL_TARGET \
auto operator op(const vector_impl& rhs) const { \
  return helper_name(indices(), rhs); \
}

#define HIPSYCL_DEFINE_UNARY_OPERATOR(helper_name, return_type, op) \
HIPSYCL_UNIVERSAL_TARGET \
auto operator op() const { \
  return helper_name(indices()); \
} \
private: \
template<int... Indices> \
HIPSYCL_UNIVERSAL_TARGET \
auto helper_name(vector_index_sequence<Indices...>) const { \
  vector_impl<return_type, dimension> result; \
  auto dummy_initializer = { \
    ((result.set<Indices>(static_cast<return_type>(get<Indices>()))), 0)... \
  }; \
  return result; \
} \
public:

template<class T, int N>
struct vector_impl
{
};

template<class T>
struct vector_impl<T,1>
{
  using indices = vector_index_sequence<0>;
  using lo_indices = vector_index_sequence<0>;
  using hi_indices = vector_index_sequence<0>;
  using even_indices = vector_index_sequence<0>;
  using odd_indices = vector_index_sequence<0>;

  static constexpr int dimension = 1;

  static_assert (intrinsic_vector<T,1>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    static_assert (component < 1, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,1,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    static_assert (component < 1, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,1,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  void transform(Function f)
  {
    data.x = f(data.x);
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    return result;
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(^=)

  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)


  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR1(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR1(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR1(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR1(>>=)


  typename intrinsic_vector<T,1>::type data;
};

template<class T>
struct vector_impl<T,2>
{
  using indices = vector_index_sequence<0,1>;

  using lo_indices = vector_index_sequence<0>;
  using hi_indices = vector_index_sequence<1>;
  using even_indices = vector_index_sequence<0>;
  using odd_indices = vector_index_sequence<1>;

  static constexpr int dimension = 2;

  static_assert (intrinsic_vector<T,2>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    static_assert (component < 2, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,2,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    static_assert (component < 2, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,2,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    result.data.y = f(data.y, other.data.y);
    return result;
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(^=)

  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)

  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR2(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR2(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR2(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR2(>>=)

  typename intrinsic_vector<T,2>::type data;
};

template<class T>
struct vector_impl<T,3>
{
  using indices = vector_index_sequence<0,1,2>;

  using lo_indices = vector_index_sequence<0,1>;
  using hi_indices = vector_index_sequence<2>;
  using even_indices = vector_index_sequence<0,2>;
  using odd_indices = vector_index_sequence<1>;

  static constexpr int dimension = 3;

  static_assert (intrinsic_vector<T,3>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    static_assert (component < 3, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,3,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    static_assert (component < 3, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,3,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
    data.z = f(data.z);
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  vector_impl binary_operation(Function f, const vector_impl& other) const
  {
    vector_impl result;
    result.data.x = f(data.x, other.data.x);
    result.data.y = f(data.y, other.data.y);
    result.data.z = f(data.z, other.data.z);
    return result;
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(^=)

  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)

  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR3(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR3(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR3(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR3(>>=)

  typename intrinsic_vector<T,3>::type data;
};

template<class T>
struct vector_impl<T,4>
{
  using indices = vector_index_sequence<0,1,2,3>;

  using lo_indices = vector_index_sequence<0,1>;
  using hi_indices = vector_index_sequence<2,3>;
  using even_indices = vector_index_sequence<0,2>;
  using odd_indices = vector_index_sequence<1,3>;

  static constexpr int dimension = 4;

  static_assert (intrinsic_vector<T,4>::exists,
                 "Required intrinsic type does not exist");

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    static_assert (component < 4, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,4,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    static_assert (component < 4, "Component must be smaller than "
                                  "the number of components");
    return vector_accessor<T,4,component>::get(data);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
  void transform(Function f)
  {
    data.x = f(data.x);
    data.y = f(data.y);
    data.z = f(data.z);
    data.w = f(data.w);
  }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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
  HIPSYCL_UNIVERSAL_TARGET
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

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(^=)

  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)

  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR4(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR4(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR4(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR4(>>=)

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
  HIPSYCL_UNIVERSAL_TARGET           \
  static T& get(Vector_type& v) \
  { return v.value; }           \
                                \
  template<class Vector_type>   \
  HIPSYCL_UNIVERSAL_TARGET           \
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
  using indices = vector_index_sequence<0,1,2,3,
                                        4,5,6,7>;

  using lo_indices = vector_index_sequence<0,1,2,3>;
  using hi_indices = vector_index_sequence<4,5,6,7>;
  using even_indices = vector_index_sequence<0,2,4,6>;
  using odd_indices = vector_index_sequence<1,3,5,7>;
  static constexpr int dimension = 8;

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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
  HIPSYCL_UNIVERSAL_TARGET
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
  HIPSYCL_UNIVERSAL_TARGET
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

  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(^=)


  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)

  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR8(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR8(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR8(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR8(>>=)

  typename intrinsic_vector<T,4>::type data0;
  typename intrinsic_vector<T,4>::type data1;
};

template<class T>
struct vector_impl<T,16>
{

  using indices = vector_index_sequence<0,1,2,3,
                                        4,5,6,7,
                                        8,9,10,11,
                                        12,13,14,15>;

  using lo_indices = vector_index_sequence<0,1,2,3,4,5,6,7>;
  using hi_indices = vector_index_sequence<8,9,10,11,12,13,14,15>;
  using even_indices = vector_index_sequence<0,2,4,6,8,10,12,14>;
  using odd_indices = vector_index_sequence<1,3,5,7,9,11,13,15>;

  static constexpr int dimension = 16;

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T& get()
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  T get() const
  {
    return vector_multi_accessor<T, component>::get(*this);
  }

  template<int component>
  HIPSYCL_UNIVERSAL_TARGET
  void set(const T& x)
  { get<component>() = x; }

  template<class Function>
  HIPSYCL_UNIVERSAL_TARGET
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
  HIPSYCL_UNIVERSAL_TARGET
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
  HIPSYCL_UNIVERSAL_TARGET
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


  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(+=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(-=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(*=)
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(/=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(%=)

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

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(^=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(&=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(|=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(^=)

  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_and, &&)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(logical_or, ||)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than, <)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(less_than_equal, <=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than, >)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(greater_than_equal, >=)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(equals, ==)
  HIPSYCL_DEFINE_BINARY_LOGICAL_OPERATOR(not_equals, !=)

  HIPSYCL_DEFINE_UNARY_OPERATOR(logical_not,
                                typename logical_vector_op_result<T>::type, !)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_UNARY_OPERATOR(bitwise_not, T, ~)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(<<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_OPERATOR16(>>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, <<)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_VECTOR_SCALAR_OPERATOR16(T, >>)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(<<=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_INPLACE_OPERATOR16(>>=)

  template<class t = T,
           std::enable_if_t<std::is_integral<t>::value>* = nullptr>
  HIPSYCL_DEFINE_BINARY_COMPONENTWISE_SCALAR_INPLACE_OPERATOR16(>>=)

  typename intrinsic_vector<T,4>::type data0;
  typename intrinsic_vector<T,4>::type data1;
  typename intrinsic_vector<T,4>::type data2;
  typename intrinsic_vector<T,4>::type data3;
};

template<class T>
struct vector_traits
{
  static constexpr bool is_scalar = true;
  static constexpr int dimension = 1;
};

template<class T, int N>
struct vector_traits<vector_impl<T,N>>
{
  static constexpr bool is_scalar = false;
  static constexpr int dimension = N;
};

}
}
}

#endif
